import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



MEDICAL_SKIN = [
    "a clinical photograph of a skin lesion",
    "a dermatoscopic image of a skin lesion",
    "a medical image of melanoma",
    "a close-up medical image of a mole",
    "a dermatology lesion image"
]

NORMAL_SKIN = [
    "normal human skin",
    "a photo of a human arm",
    "a photo of a hand",
    "a photo of a face",
    "healthy human skin"
]

NON_SKIN = [
    "an animal",
    "food",
    "a random object",
    "a landscape",
    "clothing",
    "a medical scan not related to skin"
]

def is_skin_lesion(image_path, min_margin=0.05, min_medical_score=0.10):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    texts = MEDICAL_SKIN + NORMAL_SKIN + NON_SKIN
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        similarities = (image_features @ text_features.T).softmax(dim=-1)[0]

    medical_score = similarities[:len(MEDICAL_SKIN)].mean().item()
    normal_score = similarities[len(MEDICAL_SKIN):len(MEDICAL_SKIN)+len(NORMAL_SKIN)].mean().item()
    nonskin_score = similarities[-len(NON_SKIN):].mean().item()

    margin_vs_normal = medical_score - normal_score
    margin_vs_nonskin = medical_score - nonskin_score

    # Debug print for scores
    print(
        "CLIP scores → medical: {:.3f}, normal: {:.3f} (Δ {:.3f}), nonskin: {:.3f} (Δ {:.3f})".format(
            medical_score,
            normal_score,
            margin_vs_normal,
            nonskin_score,
            margin_vs_nonskin,
        )
    )

    # Accept only if medical score is sufficiently high and ahead of the other categories by margin
    return (
        medical_score >= min_medical_score and
        margin_vs_normal >= min_margin and
        margin_vs_nonskin >= min_margin
    )
