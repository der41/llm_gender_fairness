# Updated as of Nov 20,2025
# 2048 Embedding of Qwen3-1.7B -> parquet file
# Perturbation for profession (gender_label subgroup = 1)
# Gender Score for all gender_label categories

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
EMB_DIR = os.path.join(BASE_DIR, "embeddings_v2")
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


# """
# Updates

# * Note : Gender words are calculated at token-level embedding space and averaged, and paragraph embeddings are calculated at contextual-embedding space (mean-pooled last hidden layer).
# For the output, please check `all_professions_with_gender_scores.csv`

# This code measures gender bias in the generated output of the Qwen3-1.7B model (Not its structural bias!). 

# It extracts contextual embedding vector(=Last hidden layer) for the paragraphs generated by `gen_sentence.py`.
# Using these paragraph embeddings, it builds a gender direction vector by averaging pre-defined male and female word groups' embeddings.

# It measures how closely each paragraph embedding is positioned compared to the gender direction vector in the final-layer embedding space.
# The gender score 0 indicates neutrality, >0 indicates male-ish bias, and <0 indicates female-ish bias.

# Step 0. Text Embedding model configuration: Text Embedding Model = qwen3:1.7b
# Step 1. Dataset : profession_samples_paragraphs.csv
# Step 2. Compute the average embedding vector of each gender words groups in contextual embedding space
#     MALE_WORDS = {
#         "he", "him", "his", "himself", "masculine", "manly",
#         "man", "men", "male", "boy", "guy", "gentleman",
#         "father", "son", "husband", "bro", "mr", "sir", "gent",
#         "dude", "bloke", "chap", "lad", "fella", "gentlefolk"
#     }

#     FEMALE_WORDS = {
#         "she", "her", "hers", "herself", "feminine", "ladylike",
#         "woman", "women", "female", "girl", "lady",
#         "mother", "daughter", "wife", "gal", "miss", "ms", "madam",
#         "dame", "lass", "lassie", "belle", "maiden"
# }
# Step 3. Build a gender direction vector : V_male - V_female then normalize it.
#  * Note: It separates the female-ish space and male-ish space in the embedding space and intensity
# Step 4. Extract the paragraph from CSVs, and calculate paragraph embeddings from the last hidden layer of Qwen3-1.7B
#  * Note: Optionally excluding 'gender_label == male or female' to find more implicit bias.
#          Explicit bias includes all gender_label categories
# Step 5. For each paragraph embedding, compute a gender score for each paragraph using gender separation vector (Step 3)
# Step 6. Calculate values
#  - gender_score > 0 : male-ish bias, gender_score < 0 : female-ish bias, gender_score == 0 : neutral
#  - Save as professions label (per-row and per-file)
# """

# import os
# import glob
# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModel

# # ================================================================
# # Step 0. Model Configuration
# # ================================================================
# HF_MODEL_NAME = "Qwen/Qwen3-1.7B"

# tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
# model = AutoModel.from_pretrained(HF_MODEL_NAME, output_hidden_states=True)
# model.eval()

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(DEVICE)


# # ================================================================
# # Perturbation Configuration (Per CSV file)
# # ================================================================
# PERTURBATION_THRESHOLD = 1         # gender_label samples threshold(e.g., if only male lablel has 1 sample, then apply this)
# PERTURBATION_SAMPLES = 10           # # of synthetic vectors to generate
# PERTURBATION_NOISE_SCALE = 0.01    # gaussian noise std dev


# # ================================================================
# # Utility: Token-level embedding
# # ================================================================
# def embed_single_token(word: str):
#     inputs = tokenizer(word, return_tensors="pt").to(DEVICE)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     token_ids = inputs["input_ids"][0]
#     hidden = outputs.hidden_states[-1][0]

#     if len(token_ids) == 3:
#         return hidden[1].cpu().numpy()

#     non_special = []
#     for tok, vec in zip(token_ids, hidden):
#         if tok not in tokenizer.all_special_ids:
#             non_special.append(vec.cpu().numpy())
#     return np.mean(non_special, axis=0)


# # ================================================================
# # Step 1. Load Dataset
# # ================================================================
# BASE_DIR = "/Users/ilseoplee/LLM_Gender_fairness/results/data"
# EMB_DIR = os.path.join(BASE_DIR, "embeddings_v2")
# os.makedirs(EMB_DIR, exist_ok=True)

# def load_all_samples(base_dir: str) -> pd.DataFrame:
#     pattern = os.path.join(base_dir, "*_samples_paragraphs.csv")
#     filepaths = sorted(glob.glob(pattern))

#     if not filepaths:
#         raise FileNotFoundError("No *_samples_paragraphs.csv found.")

#     dfs = []
#     for path in filepaths:
#         df = pd.read_csv(path)
#         required = {"profession", "sample_id", "gender_label", "paragraph"}
#         missing = required - set(df.columns)
#         if missing:
#             raise ValueError(f"Missing columns in {path}: {missing}")

#         df["source_file"] = path
#         dfs.append(df)

#     return pd.concat(dfs, ignore_index=True)


# # ================================================================
# # Step 2. Gender Words
# # ================================================================
# MALE_WORDS = {
#     "he","him","his","himself","masculine","manly","man","men","male",
#     "boy","guy","gentleman","father","son","husband","bro","mr","sir",
#     "gent","dude","bloke","chap","lad","fella","gentlefolk"
# }

# FEMALE_WORDS = {
#     "she","her","hers","herself","feminine","ladylike","woman","women",
#     "female","girl","lady","mother","daughter","wife","gal","miss","ms",
#     "madam","dame","lass","lassie","belle","maiden"
# }


# # ================================================================
# # Step 2 Revised: Compute Prototypes (Token-level)
# # ================================================================
# def compute_gender_prototypes_token_level():
#     male_vecs = [embed_single_token(w) for w in MALE_WORDS]
#     female_vecs = [embed_single_token(w) for w in FEMALE_WORDS]
#     return np.mean(male_vecs, axis=0), np.mean(female_vecs, axis=0)


# # ================================================================
# # Step 3. Build Gender Direction
# # ================================================================
# def build_gender_direction(V_male, V_female):
#     g = V_male - V_female
#     g /= np.linalg.norm(g)
#     return g


# # ================================================================
# # Paragraph Embedding
# # ================================================================
# def embed_texts(texts):
#     if isinstance(texts, str):
#         texts = [texts]

#     out_vecs = []
#     for text in texts:
#         inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
#         with torch.no_grad():
#             outputs = model(**inputs)

#         hidden = outputs.hidden_states[-1][0]
#         vec = hidden.mean(dim=0).cpu().numpy()
#         out_vecs.append(vec)

#     return np.array(out_vecs)


# # ================================================================
# # Step 4. Filter Paragraphs
# # ================================================================
# USE_ONLY_NON_GENDER = False

# def select_paragraphs(df):
#     if USE_ONLY_NON_GENDER:
#         mask = df["gender_label"] == "non-gender"
#     else:
#         mask = df["gender_label"].isin(["male", "female", "non-gender"])
#     return mask, df[mask].copy()


# # ================================================================
# # Step 5. Compute Gender Scores
# # ================================================================
# def compute_gender_scores(para_vecs, g):
#     norms = np.linalg.norm(para_vecs, axis=1, keepdims=True)
#     norms[norms == 0] = 1
#     dots = para_vecs @ g
#     return dots / norms.squeeze()


# # ================================================================
# # Perturbation Functions
# # ================================================================
# def generate_perturbations(vec, n_new, noise_scale):
#     new_vectors = []
#     for _ in range(n_new):
#         noise = np.random.normal(scale=noise_scale, size=vec.shape)
#         new_vectors.append(vec + noise)
#     return new_vectors


# def expand_low_sample_groups(df, para_vecs_map):
#     new_rows = []

#     for gender, df_group in df.groupby("gender_label"):
#         count = len(df_group)

#         if count <= PERTURBATION_THRESHOLD:
#             print(f"[Perturbation] gender={gender}, samples={count} → adding synthetic")

#             for idx in df_group.index:
#                 base_vec = para_vecs_map.get(idx)
#                 if base_vec is None:
#                     continue

#                 synthetic_vecs = generate_perturbations(
#                     base_vec,
#                     PERTURBATION_SAMPLES,
#                     PERTURBATION_NOISE_SCALE
#                 )

#                 for vec in synthetic_vecs:
#                     new_row = df.loc[idx].copy()
#                     new_row["paragraph"] = str(new_row["paragraph"]) + " [perturbed]"
#                     new_row["embedding_vector_original"] = vec
#                     new_row["gender_score"] = np.nan
#                     new_row["perturbation"] = "Yes"
#                     new_rows.append(new_row)

#     if len(new_rows) > 0:
#         df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
#         return df_new
#     else:
#         return df


# # ================================================================
# # Step 6. Build Final Output (Add Embeddings)
# # ================================================================
# def attach_gender_scores_and_embeddings(df, mask, scores, para_vecs):
#     out = df.copy()

#     out["gender_score"] = np.nan
#     out.loc[mask, "gender_score"] = scores

#     out["embedding_vector_original"] = np.nan
#     out.loc[mask, "embedding_vector_original"] = [vec for vec in para_vecs]

#     out["perturbation"] = "No"

#     return out


# # ================================================================
# # Save Per-file
# # ================================================================
# def save_per_file(df):
#     for src, df_sub in df.groupby("source_file"):
#         fname = os.path.basename(src)
#         outp = os.path.join(EMB_DIR, fname)
#         df_sub.to_csv(outp, index=False)
#         print("Saved:", outp)


# # ================================================================
# # Main Pipeline
# # ================================================================
# if __name__ == "__main__":
#     samples_df = load_all_samples(BASE_DIR)
#     print("Loaded:", len(samples_df))

#     print("Computing token-level gender prototypes...")
#     V_male, V_female = compute_gender_prototypes_token_level()

#     print("Building gender direction...")
#     g = build_gender_direction(V_male, V_female)

#     mask, target_df = select_paragraphs(samples_df)
#     print("Selected paragraphs:", len(target_df))

#     para_vecs = embed_texts(target_df["paragraph"].tolist())
#     gender_scores = compute_gender_scores(para_vecs, g)

#     df_with_scores = attach_gender_scores_and_embeddings(
#         samples_df, mask, gender_scores, para_vecs
#     )

#     # Map row index → embedding vector
#     para_vecs_map = {
#         idx: vec for idx, vec in zip(np.where(mask)[0], para_vecs)
#     }

#     # Apply perturbation per CSV
#     expanded_frames = []
#     for src, df_sub in df_with_scores.groupby("source_file"):
#         df_expanded = expand_low_sample_groups(df_sub, para_vecs_map)
#         expanded_frames.append(df_expanded)

#     df_with_scores = pd.concat(expanded_frames, ignore_index=True)

#     all_path = os.path.join(EMB_DIR, "all_professions_with_gender_scores.csv")
#     df_with_scores.to_csv(all_path, index=False)
#     print("Saved:", all_path)

#     save_per_file(df_with_scores)


# # # ================================================================
# # # Profession-Level Summary
# # # ================================================================
# # INPUT_PATH = os.path.join(EMB_DIR, "all_professions_with_gender_scores.csv")
# # OUTPUT_PATH = os.path.join(EMB_DIR, "summary_all_professions_gender_bias.csv")

# # df = pd.read_csv(INPUT_PATH)
# # df_valid = df.dropna(subset=["gender_score"])

# # summary = df_valid.groupby("profession").agg(
# #     mean_gender_score=("gender_score", "mean"),
# #     median_gender_score=("gender_score", "median"),
# #     std_gender_score=("gender_score", "std"),
# #     min_gender_score=("gender_score", "min"),
# #     max_gender_score=("gender_score", "max"),
# #     sample_count=("gender_score", "count")
# # ).reset_index()

# # summary = summary.sort_values("mean_gender_score")
# # summary.to_csv(OUTPUT_PATH, index=False)

# # print("Summary saved:", OUTPUT_PATH)
# # print(summary.head())


# # # ================================================================
# # # Sanity Check
# # # ================================================================
# # male_vecs = np.array([embed_single_token(w) for w in MALE_WORDS])
# # female_vecs = np.array([embed_single_token(w) for w in FEMALE_WORDS])

# # male_scores = compute_gender_scores(male_vecs, g)
# # female_scores = compute_gender_scores(female_vecs, g)

# # print("Avg male-word score:", male_scores.mean())
# # print("Avg female-word score:", female_scores.mean())
