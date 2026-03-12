import threading

import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_preprocess = None
_lock = threading.Lock()


def get_clip_model():
    """Load CLIP model once and reuse it across calls."""
    global _model, _preprocess
    if _model is None:
        with _lock:
            if _model is None:
                print("Loading CLIP model (ViT-B/32)...")
                _model, _preprocess = clip.load("ViT-B/32", device=device)
                print("CLIP model loaded")
    return _model, _preprocess
