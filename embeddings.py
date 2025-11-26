'''
This script imports description about professions, generates paragraph embeddings using Qwen3-1.7B, 
constructs a gender direction from male/female word prototypes, computes gender scores for each paragraph, 
and performs perturbation for the very few samples. 
It stores all embeddings and gender scores in Parquet and CSV files.
*Note:
Compute the average embedding vector of each gender words groups in contextual embedding space
    MALE_WORDS = {
        "he", "him", "his", "himself", "masculine", "manly",
        "man", "men", "male", "boy", "guy", "gentleman",
        "father", "son", "husband", "bro", "mr", "sir", "gent",
        "dude", "bloke", "chap", "lad", "fella", "gentlefolk"
    }

    FEMALE_WORDS = {
        "she", "her", "hers", "herself", "feminine", "ladylike",
        "woman", "women", "female", "girl", "lady",
        "mother", "daughter", "wife", "gal", "miss", "ms", "madam",
        "dame", "lass", "lassie", "belle", "maiden"
}
Build a gender direction vector : V_male - V_female then normalize it to separate the female-ish space 
and male-ish space in the embedding space and intensity.
gender_score > 0 : male-ish bias, gender_score < 0 : female-ish bias, gender_score == 0 : neutral
'''

import os
import glob
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModel

# ================================================================
# CONFIG
# ================================================================
HF_MODEL_NAME = "Qwen/Qwen3-1.7B"
BASE_DIR = "/Users/ilseoplee/LLM_Gender_fairness/results/data"
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# ---- Choose Mode ----
USE_SAMPLE_MODE = False     # True → test only 3 professions
SAMPLE_PROFESSIONS = {"artist", "scientist", "farmer"} # Test Passed! 

# ---- Perturbation ----
PERTURBATION_THRESHOLD = 1
PERTURBATION_SAMPLES = 10
PERTURBATION_NOISE_SCALE = 0.01


# ================================================================
# MODEL LOAD
# ================================================================
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModel.from_pretrained(HF_MODEL_NAME, output_hidden_states=True)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)


# ================================================================
# Helper: token-level embedding
# ================================================================
def embed_single_token(word: str):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    token_ids = inputs["input_ids"][0]
    hidden = outputs.hidden_states[-1][0]

    if len(token_ids) == 3:
        return hidden[1].cpu().numpy()

    non_special = [vec.cpu().numpy()
                   for tok, vec in zip(token_ids, hidden)
                   if tok not in tokenizer.all_special_ids]

    return np.mean(non_special, axis=0)


# ================================================================
# Step 1. Load Dataset
# ================================================================
def load_all_samples(base_dir):
    pattern = os.path.join(base_dir, "*_samples_paragraphs.csv")
    filepaths = sorted(glob.glob(pattern))

    dfs = []
    for path in filepaths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)

        # Sample-mode filter
        if USE_SAMPLE_MODE:
            prof = df["profession"].iloc[0]
            if prof not in SAMPLE_PROFESSIONS:
                continue

        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No matching profession files found!")

    return pd.concat(dfs, ignore_index=True)


# ================================================================
# Step 2. Gender Words
# ================================================================
MALE_WORDS = {
    "he","him","his","himself","masculine","manly","man","men","male",
    "boy","guy","gentleman","father","son","husband","bro","mr","sir",
    "gent","dude","bloke","chap","lad","fella","gentlefolk"
}

FEMALE_WORDS = {
    "she","her","hers","herself","feminine","ladylike","woman","women",
    "female","girl","lady","mother","daughter","wife","gal","miss","ms",
    "madam","dame","lass","lassie","belle","maiden"
}


def compute_gender_prototypes_token_level():
    male_vecs = [embed_single_token(w) for w in MALE_WORDS]
    female_vecs = [embed_single_token(w) for w in FEMALE_WORDS]
    return np.mean(male_vecs, axis=0), np.mean(female_vecs, axis=0)


# ================================================================
# Step 3. Gender Direction
# ================================================================
def build_gender_direction(V_male, V_female):
    g = V_male - V_female
    return g / np.linalg.norm(g)


# ================================================================
# Step 4. Paragraph Embedding
# ================================================================
def embed_texts(texts):
    vectors = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.hidden_states[-1][0]
        vectors.append(hidden.mean(dim=0).cpu().numpy())
    return np.array(vectors)


# ================================================================
# Step 5. Gender Score
# ================================================================
def compute_gender_scores(embeddings, g):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (embeddings @ g) / norms.squeeze()


# ================================================================
# Step 6. Profession-level Perturbation
# ================================================================
def generate_perturbations(vec, count, noise_scale):
    return [vec + np.random.normal(scale=noise_scale, size=vec.shape) for _ in range(count)]


def expand_low_sample_groups(df):
    df = df.copy()
    new_rows = []

    # Profession-level perturbation
    for prof, df_prof in df.groupby("profession"):

        for gender, df_group in df_prof.groupby("gender_label"):

            real_count = len(df_group)

            if real_count <= PERTURBATION_THRESHOLD:
                print(f"[Perturbation] profession={prof}, gender={gender}, samples={real_count}")

                for idx in df_group.index:
                    base_vec = df.loc[idx, "embedding_vector_original"]

                    synthetic_vecs = generate_perturbations(
                        base_vec,
                        PERTURBATION_SAMPLES,
                        PERTURBATION_NOISE_SCALE
                    )

                    for vec in synthetic_vecs:
                        new_row = df.loc[idx].copy()
                        new_row["embedding_vector_original"] = vec
                        new_row["paragraph"] += " [perturbed]"
                        new_row["perturbation"] = "Yes"
                        new_rows.append(new_row)

    if new_rows:
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


# ================================================================
# MAIN PIPELINE
# ================================================================
if __name__ == "__main__":

    samples_df = load_all_samples(BASE_DIR)
    print("Loaded Rows:", len(samples_df))

    # Gender axis
    V_male, V_female = compute_gender_prototypes_token_level()
    g = build_gender_direction(V_male, V_female)

    # Original paragraph embeddings
    embeddings = embed_texts(samples_df["paragraph"].tolist())
    samples_df["embedding_vector_original"] = list(embeddings)
    samples_df["perturbation"] = "No"

    # Profession-level synthetic augmentation
    expanded_df = expand_low_sample_groups(samples_df)

    # Recompute gender score for all rows (real + synthetic)
    all_embeddings = np.vstack(expanded_df["embedding_vector_original"].values)
    expanded_df["gender_score"] = compute_gender_scores(all_embeddings, g)

    # -----------------------
    # Save Parquet
    # -----------------------
    table = pa.Table.from_pydict({
        "profession": expanded_df["profession"],
        "sample_id": expanded_df["sample_id"],
        "question": expanded_df["question"],
        "gender_label": expanded_df["gender_label"],
        "paragraph": expanded_df["paragraph"],
        "adjectives": expanded_df["adjectives"],
        "source_file": expanded_df["source_file"],
        "perturbation": expanded_df["perturbation"],
        "gender_score": expanded_df["gender_score"],
        "embedding_vector": expanded_df["embedding_vector_original"].tolist(),
    })

    pq_path = os.path.join(EMB_DIR, "all_embeddings.parquet")
    pq.write_table(table, pq_path)
    print("Saved Parquet:", pq_path)

    # -----------------------
    # Save summary CSV
    # -----------------------
    csv_df = expanded_df.drop(columns=["embedding_vector_original"])
    csv_path = os.path.join(EMB_DIR, "all_professions_with_gender_scores.csv")
    csv_df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)
