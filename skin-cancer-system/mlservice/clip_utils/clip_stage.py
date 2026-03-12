import clip
import torch
from PIL import Image
from clip_utils._clip_model import get_clip_model, device

STAGE_PROMPTS = {
    "Stage 0 (In situ : a very early melanoma confined to the epidermis)": "a very early melanoma confined to the epidermis",
    "Stage I (Thin melanoma : a small thin melanoma less than 1mm)": "a small thin melanoma less than 1mm",
    "Stage II (Thick melanoma : a thick melanoma with irregular borders)": "a thick melanoma with irregular borders",
    "Stage III (Lymph involvement : melanoma spread to nearby lymph nodes)": "melanoma spread to nearby lymph nodes",
    "Stage IV (Advanced melanoma : advanced melanoma with metastasis)": "advanced melanoma with metastasis"
}

def estimate_melanoma_stage(image_path):
    model, preprocess = get_clip_model()
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize(list(STAGE_PROMPTS.values())).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        similarity = (image_features @ text_features.T).softmax(dim=-1)

    best_idx = similarity.argmax().item()
    return list(STAGE_PROMPTS.keys())[best_idx]
