"""CLIP utility for skin lesion classification pre-filter"""
import clip
import torch
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ CLIP using device: {device}")

try:
    print("⏳ Loading CLIP model ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("✅ CLIP model loaded successfully")
except Exception as e:
    print(f"⚠️ CLIP model loading failed with error: {e}")
    import traceback
    traceback.print_exc()
    model = None
    preprocess = None

TEXT_PROMPTS = [
    "a photo of a skin lesion",
    "a medical image of melanoma",
    "a photo of human skin disease",
    "a dermatology image",
    "a close up photo of a mole on skin"
]

NEGATIVE_PROMPTS = [
    "a dog",
    "a cat",
    "a car",
    "a building",
    "a random object",
    "a landscape",
    "a person's face",
    "food",
    "a plant",
    "furniture",
    "a cartoon",
    "text or document"
]

def is_skin_lesion(image_path, threshold=0.15, gap_margin=0.02):
    """
    Check if uploaded image is a skin lesion using CLIP.

    A hit requires ALL THREE:
      - Average skin prompt score above `threshold`
      - Skin score exceeds non‑skin score by `gap_margin`
      - The highest scoring prompt must be a skin prompt

    Balanced to accept skin lesions while rejecting non-skin images.
    """
    if model is None or preprocess is None:
        print("⚠️ CLIP model not available, allowing image (will rely on CNN only)")
        return True  # Allow if CLIP fails - CNN will make the final decision
    
    try:
        # Load and preprocess image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        
        # Tokenize prompts
        texts = clip.tokenize(TEXT_PROMPTS + NEGATIVE_PROMPTS).to(device)
        
        # Compute similarity
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(texts)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity probabilities
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Positive vs negative scores
        pos_scores = similarity[0][:len(TEXT_PROMPTS)]
        neg_scores = similarity[0][len(TEXT_PROMPTS):]

        skin_score = pos_scores.mean().item()
        non_skin_score = neg_scores.mean().item()
        score_gap = skin_score - non_skin_score

        # Strongest class must be a skin prompt
        max_idx = similarity[0].argmax().item()
        max_is_skin = max_idx < len(TEXT_PROMPTS)

        print(
            f"🔍 CLIP scores → skin: {skin_score:.4f}, non-skin: {non_skin_score:.4f}, gap: {score_gap:.4f}, max_is_skin: {max_is_skin} | "
            f"needs skin>{threshold} AND gap>{gap_margin} AND max_is_skin"
        )
        
        is_skin = (skin_score > threshold) and (score_gap > gap_margin) and max_is_skin
        return is_skin
    
    except Exception as e:
        print(f"⚠️ CLIP check failed: {e}")
        return True  # Allow if check fails - CNN will handle it


def melanoma_stage(confidence):
    """
    Map CNN confidence score to a simple three-tier stage.

    NOTE: This is a heuristic mapping. It is not a clinical staging system.
    """
    if confidence < 0.55:
        return "Early Stage"
    if confidence < 0.75:
        return "Intermediate Stage"
    return "Advanced Stage"
