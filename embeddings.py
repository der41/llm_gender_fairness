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
from tqdm import tqdm  # pip install tqdm if you don't have it

# ================================================================
# CONFIG
# ================================================================
HF_MODEL_NAME = "Qwen/Qwen3-1.7B"
BASE_DIR = "/Users/diegorodriguezescalona/Documents/Duke/Classes/3-Fall_2025/XAI/LLM_Gender_fairness/results/data"
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# ---- Performance Config ----
BATCH_SIZE = 32  # Increase to 64 or 128 if you have a lot of VRAM/RAM

# ---- Choose Mode ----
USE_SAMPLE_MODE = False
SAMPLE_PROFESSIONS = {"artist", "scientist", "farmer"}

# ---- Perturbation ----
PERTURBATION_THRESHOLD = 1
PERTURBATION_SAMPLES = 10
PERTURBATION_NOISE_SCALE = 0.01


# ================================================================
# MODEL LOAD & DEVICE SETUP
# ================================================================
print("Loading Model...")

# 1. Smart Device Detection
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(">> Using Device: CUDA (Nvidia)")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print(">> Using Device: MPS (Mac Silicon)")
else:
    DEVICE = "cpu"
    print(">> Using Device: CPU (Warning: Slow)")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
model = AutoModel.from_pretrained(HF_MODEL_NAME, output_hidden_states=True)
model.eval()
model.to(DEVICE)


# ================================================================
# OPTIMIZED: Batched Embedding Function
# ================================================================
def get_embeddings_batched(text_list, batch_size=32):
    """
    Computes embeddings for a list of texts using batch processing.
    Handles padding and attention masks automatically.
    """
    all_embeddings = []
    
    # Process in chunks with a progress bar
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Extract last hidden state
        hidden_states = outputs.hidden_states[-1] # (Batch, Seq_Len, Dim)
        
        # Create a mask to ignore padding tokens in the average
        # unsqueeze(-1) makes mask shape (Batch, Seq_Len, 1) to broadcast over Dim
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        
        # Zero out padding vectors
        masked_hidden = hidden_states * attention_mask
        
        # Sum vectors along sequence length and divide by actual token count
        sum_embeddings = torch.sum(masked_hidden, dim=1)
        token_counts = torch.sum(attention_mask, dim=1).clamp(min=1e-9) # Avoid div by zero
        
        mean_embeddings = sum_embeddings / token_counts
        
        all_embeddings.append(mean_embeddings.cpu().numpy())
        
    if len(all_embeddings) > 0:
        return np.vstack(all_embeddings)
    return np.array([])


# ================================================================
# Step 1. Load Dataset
# ================================================================
def load_all_samples(base_dir):
    pattern = os.path.join(base_dir, "*_samples_paragraphs.csv")
    filepaths = sorted(glob.glob(pattern))

    dfs = []
    print(f"Found {len(filepaths)} files. Loading...")
    
    for path in filepaths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)

        if USE_SAMPLE_MODE:
            prof = df["profession"].iloc[0]
            if prof not in SAMPLE_PROFESSIONS:
                continue

        dfs.append(df)

    if not dfs:
        raise ValueError("No matching profession files found!")

    return pd.concat(dfs, ignore_index=True)


# ================================================================
# Step 2 & 3. Optimized Gender Direction
# ================================================================
MALE_WORDS = [
    "he","him","his","himself","masculine","manly","man","men","male",
    "boy","guy","gentleman","father","son","husband","bro","mr","sir",
    "gent","dude","bloke","chap","lad","fella","gentlefolk"
]

FEMALE_WORDS = [
    "she","her","hers","herself","feminine","ladylike","woman","women",
    "female","girl","lady","mother","daughter","wife","gal","miss","ms",
    "madam","dame","lass","lassie","belle","maiden"
]

def compute_gender_direction_batched():
    print("Computing Gender Prototypes...")
    # Process all male words in one batch
    male_embeddings = get_embeddings_batched(list(MALE_WORDS), batch_size=len(MALE_WORDS))
    V_male = np.mean(male_embeddings, axis=0)

    # Process all female words in one batch
    female_embeddings = get_embeddings_batched(list(FEMALE_WORDS), batch_size=len(FEMALE_WORDS))
    V_female = np.mean(female_embeddings, axis=0)

    # Compute direction
    g = V_male - V_female
    return g / np.linalg.norm(g)


# ================================================================
# Step 5. Gender Score (Vectorized)
# ================================================================
def compute_gender_scores(embeddings, g):
    # Dot product of matrix (N, Dim) and vector (Dim,) -> (N,)
    dot_products = embeddings @ g
    
    # Norms of every row
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Avoid division by zero
    norms[norms == 0] = 1
    
    return dot_products / norms


# ================================================================
# Step 6. Optimized Perturbation (NumPy Vectorization)
# ================================================================
def expand_low_sample_groups_optimized(df):
    """
    Identifies small groups and uses vectorized NumPy operations 
    to create synthetic samples without Python loops.
    """
    print("Checking for low-sample groups to perturb...")
    
    # 1. Identify groups that need perturbation
    # value_counts is fast.
    # Create a key based on profession + gender
    df['group_key'] = df['profession'].astype(str) + "_" + df['gender_label'].astype(str)
    counts = df['group_key'].value_counts()
    
    # Filter groups below threshold
    small_groups = counts[counts <= PERTURBATION_THRESHOLD].index
    
    if len(small_groups) == 0:
        return df.drop(columns=['group_key'])

    # 2. Filter the rows that belong to these small groups
    rows_to_expand = df[df['group_key'].isin(small_groups)].copy()
    
    if rows_to_expand.empty:
        return df.drop(columns=['group_key'])

    print(f">> Perturbing {len(rows_to_expand)} rows x {PERTURBATION_SAMPLES} samples...")

    # 3. Repeat rows using pandas index repeat
    # If we have 2 rows and samples=10, we get 20 rows
    expanded_rows = rows_to_expand.loc[rows_to_expand.index.repeat(PERTURBATION_SAMPLES)].copy()
    
    # 4. Generate Noise Matrix
    # Shape: (Total Expanded Rows, Embedding Dim)
    original_vecs = np.vstack(expanded_rows["embedding_vector_original"].values)
    noise = np.random.normal(scale=PERTURBATION_NOISE_SCALE, size=original_vecs.shape)
    
    # 5. Apply Noise
    perturbed_vecs = original_vecs + noise
    
    # 6. Update Metadata
    expanded_rows["embedding_vector_original"] = list(perturbed_vecs)
    expanded_rows["paragraph"] = expanded_rows["paragraph"] + " [perturbed]"
    expanded_rows["perturbation"] = "Yes"
    
    # 7. Combine
    final_df = pd.concat([df, expanded_rows], ignore_index=True)
    return final_df.drop(columns=['group_key'])


# ================================================================
# MAIN PIPELINE
# ================================================================
if __name__ == "__main__":

    # 1. Load Data
    samples_df = load_all_samples(BASE_DIR)
    print(f"Loaded {len(samples_df)} original rows.")

    # 2. Compute Gender Axis
    g = compute_gender_direction_batched()

    # 3. Compute Embeddings (Batched)
    print(f"Embedding paragraphs (Batch Size: {BATCH_SIZE})...")
    embeddings = get_embeddings_batched(samples_df["paragraph"].tolist(), batch_size=BATCH_SIZE)
    samples_df["embedding_vector_original"] = list(embeddings)
    samples_df["perturbation"] = "No"

    # 4. Apply Perturbation (Vectorized)
    expanded_df = expand_low_sample_groups_optimized(samples_df)
    print(f"Final dataset size: {len(expanded_df)} rows")

    # 5. Compute Scores (Vectorized)
    all_embeddings = np.vstack(expanded_df["embedding_vector_original"].values)
    expanded_df["gender_score"] = compute_gender_scores(all_embeddings, g)

    # -----------------------
    # Save Parquet
    # -----------------------
    print("Saving data...")
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
    print(f"Saved Parquet: {pq_path}")

    # -----------------------
    # Save CSV
    # -----------------------
    csv_df = expanded_df.drop(columns=["embedding_vector_original"])
    csv_path = os.path.join(EMB_DIR, "all_professions_with_gender_scores.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")