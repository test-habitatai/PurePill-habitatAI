import os
import torch
import numpy as np
from PIL import Image
from numpy.linalg import norm
from torchvision import models, transforms

# ------------------------------------------------------------
# 1. Load Pretrained ResNet50 Model
# ------------------------------------------------------------
try:
    MODEL = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    MODEL = torch.nn.Sequential(*list(MODEL.children())[:-1])  # remove FC layer → 2048-d embedding
    MODEL.eval()
    print("[INFO] ResNet50 loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load ResNet50: {e}")
    MODEL = None

# ------------------------------------------------------------
# 2. Transform Pipeline
# ------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------
# 3. Extract Embedding
# ------------------------------------------------------------
def extract_embedding(img_path: str):
    if MODEL is None:
        raise ValueError("Model not loaded.")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    tensor_img = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        embedding = MODEL(tensor_img).flatten().numpy()

    return embedding

# ------------------------------------------------------------
# 4. Cosine Similarity
# ------------------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# ------------------------------------------------------------
# 5. Load Reference Embeddings from saved .npy
# ------------------------------------------------------------
import os
import numpy as np

try:
    # Path to saved embeddings
    EMB_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "reference_embeddings.npy")
    )
    
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"{EMB_PATH} not found. Please run generate_embeddings.py first.")
    
    REFERENCE_EMBEDDING = np.load(EMB_PATH)
    print(f"[INFO] Reference embeddings loaded from {EMB_PATH}")
except Exception as e:
    print(f"[WARNING] Reference embeddings not loaded: {e}")
    REFERENCE_EMBEDDING = None


# ------------------------------------------------------------
# 6. Run Defect Detection
# ------------------------------------------------------------
def detect_pill_defect(image_path: str, threshold=0.50):
    """
    Detect pill defects by comparing with reference embedding.

    Returns:
        dict: {
            'similarity_score': float,
            'defect_score': float,
            'status': 'PASS' or 'FAIL'
        }
    """
    if REFERENCE_EMBEDDING is None:
        raise ValueError("Reference embedding not available.")

    test_emb = extract_embedding(image_path)
    sim = cosine_similarity(test_emb, REFERENCE_EMBEDDING)

    sim = max(0.0, min(1.0, sim))
    defect_score = (1 - sim) * 100

    status = "PASS" if sim >= threshold else "FAIL"

    return {
        "similarity_score": round(sim, 4),
        "defect_score": round(defect_score, 2),
        "status": status
    }

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
# if __name__ == "__main__":
#     test_image = os.path.join(os.path.dirname(__file__), "..", "models", "test_pills", "pill5.jpg")
#     if os.path.exists(test_image):
#         result = detect_pill_defect(test_image)
#         print(f"[RESULT] {result}")
#     else:
#         print(f"[INFO] Test image not found at {test_image}")
