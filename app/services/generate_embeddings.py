import os
import numpy as np
from pill_defect_model import extract_embedding  # Make sure your PYTHONPATH includes app/

# Path to your reference pill images
ref_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "reference_pills"))

# Collect embeddings
embeddings = []

files = [f for f in os.listdir(ref_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if len(files) == 0:
    raise ValueError("No reference pill images found in " + ref_folder)

for file in files:
    emb = extract_embedding(os.path.join(ref_folder, file))
    embeddings.append(emb)

# Average or stack embeddings
reference_embedding = np.mean(embeddings, axis=0)

# Save embeddings to a file
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "reference_embeddings.npy"))
np.save(save_path, reference_embedding)

print(f"[INFO] Reference embeddings saved to {save_path}")
