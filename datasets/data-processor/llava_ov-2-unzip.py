import os
import shutil
import tarfile
from tqdm import tqdm
from pathlib import Path
import json


def convert_dataset(src_base, target_base)
    target_folders = {"cambrian", "ureader_kg", "ureader_qa"}
    folders = [
        f for f in os.listdir(src_base)
        if os.path.isdir(os.path.join(src_base, f)) and any(t in f for t in target_folders)
    ]

    for folder in folders:
        src_folder = os.path.join(src_base, folder)
        dst_folder = os.path.join(target_base, folder)
        
        os.makedirs(dst_folder, exist_ok=True)
        
        for file in os.listdir(src_folder):
            src_path = os.path.join(src_folder, file)
            
            if file.endswith('.json'):
                with open(src_path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    print(f"Processing {file} with {len(data)} items...")
                    for item in tqdm(data, desc=f"Updating image paths in {file}"):
                        if 'image' in item:
                            original_path = item['image']
                            root = os.path.basename(target_base)
                            new_path = f"{root}/{folder}/image/{original_path}"
                            item['image'] = new_path
                else:
                    print(f"Warning: {file} is not a list. Skipping...")
                
                dst_json_path = os.path.join(target_base, folder, file)
                with open(dst_json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Processed and saved {file} to {target_base}")
                
            elif 'images_' in file and file.endswith('.tar.gz'):
                image_dir = os.path.join(dst_folder, "image")
                os.makedirs(image_dir, exist_ok=True)
                
                with tarfile.open(src_path, 'r:gz') as tar:
                    tar.extractall(path=image_dir)
                print(f"Extracted {file} to {image_dir}")

    print("All files processed successfully!")



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--source-root", default="datasets/OneVisionData",)
    parser.add_argument("--target-root", default="datasets/OneVisionData",)
    args = parser.parse_args()

    args.source_root = os.path.abspath(args.source_root)
    args.target_root = os.path.abspath(args.target_root)

    os.makedirs(args.target_root, exist_ok=True)

    convert_dataset(args.source_root, args.target_root)


if __name__ == "__main__":
    main()