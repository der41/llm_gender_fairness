import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from numpy.linalg import norm

# --- CONFIGURATION ---
# Ensure these paths match your local machine
BASE_DIR = "results/data/embeddings"
EMB_PATH = os.path.join(BASE_DIR, "all_embeddings.parquet")
CENTROID_PATH = os.path.join(BASE_DIR, "profession_centroid_vectors.parquet")
OUTPUT_FILE = "fairness-lens/public/results/data/pca_circles.json"

def to_np(v):
    if v is None: return None
    try: return np.array(v, dtype=float)
    except: return None

def process_data():
    if not os.path.exists(EMB_PATH) or not os.path.exists(CENTROID_PATH):
        print(f"Error: Could not find parquet files in {BASE_DIR}")
        return

    print("Loading data...")
    df_emb = pq.read_table(EMB_PATH).to_pandas()
    df_emb["embedding_vector"] = df_emb["embedding_vector"].apply(lambda x: np.array(x, dtype=float))
    
    df_cent = pd.read_parquet(CENTROID_PATH)
    df_cent["centroid_male"] = df_cent["centroid_male"].apply(to_np)
    df_cent["centroid_female"] = df_cent["centroid_female"].apply(to_np)
    df_cent["centroid_nongender"] = df_cent["centroid_nongender"].apply(to_np)

    professions = df_emb["profession"].unique()
    output_data = {}

    for profession in professions:
        print(f"Processing {profession}...")
        
        # Filter Data
        df_prof = df_emb[df_emb["profession"] == profession]
        cent_row = df_cent[df_cent["profession"] == profession]
        
        if df_prof.empty or cent_row.empty: continue
        
        cent = cent_row.iloc[0]

        # Prepare Vectors for PCA
        groups = {
            "male": df_prof[df_prof["gender_label"] == "male"]["embedding_vector"].tolist(),
            "female": df_prof[df_prof["gender_label"] == "female"]["embedding_vector"].tolist(),
            "neutral": df_prof[df_prof["gender_label"] == "non-gender"]["embedding_vector"].tolist() # Renamed to neutral for app consistency
        }
        
        centroids_map = {
            "male": cent["centroid_male"],
            "female": cent["centroid_female"],
            "neutral": cent["centroid_nongender"]
        }

        all_vecs = []
        # Add all points
        for label, vecs in groups.items():
            all_vecs.extend(vecs)
        # Add centroids
        for c_vec in centroids_map.values():
            if c_vec is not None: all_vecs.append(c_vec)

        if len(all_vecs) < 3: continue

        # Run PCA
        pca = PCA(n_components=2)
        pca.fit(np.stack(all_vecs))

        # Calculate Coordinates & Radius
        prof_data = {}
        for label in ["male", "female", "neutral"]:
            c_vec = centroids_map[label]
            points = groups[label]
            
            if c_vec is not None and len(points) > 0:
                # Transform centroid
                c_2d = pca.transform([c_vec])[0]
                
                # Transform points to get radius
                pts_2d = pca.transform(points)
                # 80% rule
                max_dist = np.max(norm(pts_2d - c_2d, axis=1))
                radius = max_dist * 0.8
                
                prof_data[label] = {
                    "x": float(c_2d[0]),
                    "y": float(c_2d[1]),
                    "r": float(radius)
                }
            else:
                prof_data[label] = None
        
        output_data[profession] = prof_data

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f)
    print(f"\nSuccess! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()