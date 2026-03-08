"""Extract and test the vision encoder (ViT) + projector from a trained MLLM checkpoint."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from transformers import SiglipVisionModel, SiglipImageProcessor


def extract_vision_encoder(mllm_path, device="cuda"):
    """Extract ViT + projector from a trained MLLM checkpoint.

    Args:
        mllm_path: path to the MLLM checkpoint directory
        device: torch device

    Returns:
        (vit, projector, processor) ready for inference
    """
    mllm_path = Path(mllm_path)

    # 1. Read config
    with open(mllm_path / "config.json") as f:
        config = json.load(f)
    mm_vision_tower = config["mm_vision_tower"]
    mm_hidden_size = config["mm_hidden_size"]
    hidden_size = config["hidden_size"]

    # 2. Scan safetensors for vision tower and projector weights
    want_prefixes = [
        "model.mm_projector.", "mm_projector.",
        "model.vision_tower.", "vision_tower.",
    ]
    vit_sd, proj_sd = {}, {}

    for p in mllm_path.glob("**/*.safetensors"):
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                if not any(k.startswith(pref) for pref in want_prefixes):
                    continue
                tensor = f.get_tensor(k)
                # Strip prefix
                clean = k
                for pref in want_prefixes:
                    if clean.startswith(pref):
                        clean = clean[len(pref):]
                        break
                if k.split(".")[1] == "mm_projector" or k.startswith("mm_projector."):
                    proj_sd[clean] = tensor.cpu()
                else:
                    vit_sd[clean] = tensor.cpu()

    # 3. Load base ViT and remove last encoder layer + head
    vit = SiglipVisionModel.from_pretrained(mm_vision_tower)
    processor = SiglipImageProcessor.from_pretrained(mm_vision_tower)

    del vit.vision_model.encoder.layers[-1:]
    vit.vision_model.head = nn.Identity()

    # 4. Clean keys and load ViT weights
    vit_sd = {k.replace("vision_tower.", "", 1): v for k, v in vit_sd.items()}
    vit.load_state_dict(vit_sd, strict=True)
    vit.eval().to(device)

    # 5. Build and load projector
    projector = nn.Sequential(
        nn.Linear(mm_hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    ).to(device).eval()
    projector.load_state_dict(proj_sd, strict=True)

    return vit, projector, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vision encoder from MLLM checkpoint")
    parser.add_argument(
        "--mllm_path",
        type=str,
        default="/home/user/PIVOT/checkpoints/stage2/stage2-sft-siglip2-so400m-Qwen2.5-1.5B-llava",
        help="Path to the MLLM checkpoint directory",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit, projector, processor = extract_vision_encoder(args.mllm_path, device=device)

    # Test inference
    img_path = Path(__file__).resolve().parent / "simpson.jpg"
    img = Image.open(img_path).convert("RGB")
    batch = processor.preprocess(img, return_tensors="pt")
    pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        vit_out = vit(pixel_values=pixel_values, output_hidden_states=True)
        vit_feats = vit_out.hidden_states[-1]
        proj_feats = projector(vit_feats)

    print("vit feature shape:", tuple(vit_feats.shape))
    print("projected feature shape:", tuple(proj_feats.shape))
