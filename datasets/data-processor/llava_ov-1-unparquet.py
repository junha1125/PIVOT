import os
import glob
import argparse
import json
import csv
from io import BytesIO

from datasets import load_dataset, Image as HFImage
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

def convert_dataset(current_dir: str, target_base: str):

    base_name   = os.path.basename(current_dir.rstrip("/"))
    target_root = os.path.join(target_base, base_name)
    image_dir   = os.path.join(target_root, "image")
    os.makedirs(image_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(current_dir, "*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(current_dir, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        print(f"[SKIP] '{base_name}' do not have parquet files.")
        return

    print(f"[START] Converting '{base_name}', {len(parquet_files)} files…")

    # ── 1) turn off auto-decoding & load dataset ────────────────────────────────
    ds = (
        load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            split="train",
        )
        .cast_column("image", HFImage(decode=False))  
    )

    converted, bad_samples = [], []

    for row in tqdm(ds, desc=f"  → {base_name}"):
        rec = {
            "id": row["id"],
            "conversations": row.get("conversations", []),
        }

        bytes_ = row["image"].get("bytes") if row.get("image") else None
        if bytes_:
            try:
                img = Image.open(BytesIO(bytes_))
                img.load()  

                if getattr(img, "mode", "RGB") != "RGB":
                    img = img.convert("RGB")

                img_fname = f"{row['id']}.jpg"
                rec["image"] = os.path.join("OneVisionData", base_name, "image", img_fname)
                
                save_path = os.path.join(image_dir, img_fname)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                img.save(os.path.join(image_dir, img_fname))

            except (UnidentifiedImageError, OSError) as e:
                bad_samples.append({"id": row["id"], "reason": str(e)})
                continue  

        converted.append(rec)

    # ── 2) save json file ─────────────────────────────────────────────
    json_mapping = {}
    mapping_csv = os.path.join(os.path.dirname(__file__), "OneVisionData", "JSON_Mapping.csv")
    if os.path.exists(mapping_csv):
        with open(mapping_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                json_mapping[row["folder_name"]] = row["json_file"]

    json_name = json_mapping.get(base_name, f"{base_name}.json")
    json_path = os.path.join(target_root, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    # ── 3) save log ────────────────────────────────────────
    if bad_samples:
        log_path = os.path.join(target_root, f"{base_name}_bad_samples.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "reason"])
            writer.writeheader()
            writer.writerows(bad_samples)
        print(f"[WARN] {len(bad_samples):,} bad samples logged to {log_path}")

    print(f"[DONE] '{base_name}': {len(converted):,} good samples → {json_path}")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--source-root", default="datasets/OneVisionData",)
    parser.add_argument("--target-root", default="datasets/OneVisionData",)
    args = parser.parse_args()

    args.source_root = os.path.abspath(args.source_root)
    args.target_root = os.path.abspath(args.target_root)

    os.makedirs(args.target_root, exist_ok=True)

    for entry in sorted(os.listdir(args.source_root)):
        src_path = os.path.join(args.source_root, entry)
        if entry.startswith(".") or not os.path.isdir(src_path):
            continue
        target_folder = os.path.join(args.target_root, entry)
        if os.path.exists(target_folder):
            print(f"[SKIP] '{entry}' already exists in target directory")
            continue
        convert_dataset(src_path, args.target_root)


if __name__ == "__main__":
    main()
