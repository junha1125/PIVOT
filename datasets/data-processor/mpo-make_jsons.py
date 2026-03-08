import os
import json
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--data_size", type=int, default=20_000)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_root", type=str, default="datasets/MMPR-v1.2")
    parser.add_argument("--token_limit", type=int, default=1500)
    return parser.parse_args()


def load_meta_and_file_infos(meta_path: str, dataset_root):
    meta = json.load(open(meta_path, encoding="utf-8"))

    file_infos = []
    for info in meta.values():
        ann_rel = info["annotation"]  # ex) "foo.jsonl"
        root = info["root"]           # ex) "bar"
        prefix = os.path.join(root)  
        ann_path = os.path.join(dataset_root, ann_rel)
        file_infos.append((ann_path, prefix, root))
    return meta, file_infos


def load_raw_items(file_infos):
    raw_items = []
    for ann_path, prefix, root in file_infos:
        print(f"Loading {ann_path} ...")
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # skip multiple images
                if "image" in item.keys():
                    rel_img = item["image"]
                    if isinstance(rel_img, list):
                        print(f"Skipping entire file {ann_path} due to multiple images...")
                        break
                raw_items.append((item, prefix, root))
    return raw_items


def sample_raw_items(raw_items, seed: int, data_size: int):
    print(f"Total items loaded: {len(raw_items)}")
    random.seed(seed)
    indices = list(range(len(raw_items)))
    random.shuffle(indices)
    sampled_indices = indices[:data_size * 2]
    raw_items = [raw_items[i] for i in sampled_indices]
    return raw_items


def build_sft_sample(item, prefix, root, iter, dataset_root: str):
    if "image" not in item.keys():
        conv = [
            {"from": "human", "value": item.get("question", "")},
            {"from": "gpt",   "value": item.get("chosen", "")}
        ]
        img_path = None
        _id = str(random.randint(10**7, 10**8 - 1))
        print(f"{_id} has no image, using random ID")
        return conv, img_path, _id

    rel_img = item["image"]

    if rel_img.endswith(".gif"):
        rel_img = rel_img[:-4] + ".jpg"
    assert rel_img.endswith((".jpg", ".jpeg", ".png")), f"Invalid image format: {rel_img}"
    img_path = os.path.join(prefix, rel_img)

    if not os.path.exists(os.path.join(dataset_root, img_path)):
        print(f"Image file does not exist, skipping...")
        print(f"Path: {os.path.join(dataset_root, img_path)}")
        return None

    parent = os.path.basename(os.path.dirname(img_path))
    fname = os.path.splitext(os.path.basename(img_path))[0]
    ramdom_num = str(random.randint(10**4, 10**5 - 1))
    _id = f"{parent}-{fname}-{ramdom_num}" if parent else fname

    # check human value
    human_value = item.get("question", "")
    if "<image>" not in human_value:
        human_value = f"<image>\n{human_value}"

    # check gpt value
    gpt_value = item.get("chosen", "")
    if "<iamge>" in gpt_value:
        return None

    conv = [
        {"from": "human", "value": human_value},
        {"from": "gpt",   "value": gpt_value}
    ]
    
    print(f"Built {iter}-th sample with ID {_id}")
    return conv, img_path, _id


def build_dpo_sample(item, prefix, root):
    if "image" not in item.keys():
        prompt = item.get("question", "")
        assert "<image>" not in item.get("chosen", ""), f"<image> found in {item.get('chosen', '')}"
        assert "<image>" not in item.get("rejected", ""), f"<image> found in {item.get('rejected', '')}"
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        img_path = None
        return prompt, chosen, rejected, img_path

    rel_img = item["image"]

    if rel_img.endswith(".gif"):
        rel_img = rel_img[:-4] + ".jpg"
    assert rel_img.endswith((".jpg", ".jpeg", ".png")), f"Invalid image format: {rel_img}"
    img_path = os.path.join(prefix, rel_img)

    if "<image>" not in item.get("question", ""):
        prompt = f"<image>\n{item.get('question', '')}"
    else:
        prompt = item.get("question", "")

    if "<image>" in item.get("chosen", ""):
        return None
    if "<image>" in item.get("rejected", ""):
        r = item.get("rejected", "")
        print(f"<image> found in rejected, skipping...")
        return None

    chosen = item.get("chosen", "")
    rejected = item.get("rejected", "")
    return prompt, chosen, rejected, img_path


def passes_token_limit(tokenizer, prompt: str, chosen: str, token_limit: int):
    merged = f"{prompt} {chosen}".strip()
    tokens = tokenizer.encode(merged, add_special_tokens=False)
    if len(tokens) > token_limit:
        print("Skipping due to token limit exceeded")
        return False
    return True


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    SEED = args.seed
    DATA_SIZE = args.data_size

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    base_dir = os.path.dirname(__file__)
    meta_path = os.path.join(base_dir, "MMPR-v1.2", "meta.json")

    meta, file_infos = load_meta_and_file_infos(meta_path, args.dataset_root)
    raw_items = load_raw_items(file_infos)

    random.seed(SEED)
    random.shuffle(raw_items)

    # import pdb; pdb.set_trace()

    sft_results = []
    dpo_results = []
    iter = 0

    for item, prefix, root in raw_items:
        # ------- SFT -------
        sft_built = build_sft_sample(item, prefix, root, iter, dataset_root=args.dataset_root)
        if sft_built is None:
            continue
        conv, sft_img_path, _id = sft_built

        # ------- DPO -------
        dpo_built = build_dpo_sample(item, prefix, root)
        if dpo_built is None:
            continue
        prompt, chosen, rejected, dpo_img_path = dpo_built

        assert sft_img_path == dpo_img_path
        img_path = os.path.join("MMPR-v1.2", dpo_img_path)

        # --- remove data with too long responses ---
        if not passes_token_limit(tokenizer, prompt, chosen, args.token_limit):
            continue
        # ------------------

        if img_path is None:
            sft_out = {
                "id": _id,
                "conversations": conv,
                "data_source": root
            }
            dpo_out = {
                "id": _id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        else:
            sft_out = {
                "id": _id,
                "conversations": conv,
                "data_source": root,
                "image": img_path
            }
            dpo_out = {
                "id": _id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "image": img_path
            }

        sft_results.append(sft_out)
        dpo_results.append(dpo_out)

        iter += 1
        if iter >= DATA_SIZE:
            break

    sft_out_path = os.path.join(base_dir, "MMPR-v1.2", f"sft_mmpr_{str(DATA_SIZE)}_{SEED}.json")
    save_json(sft_results, sft_out_path)
    print(f"Total SFT samples: {len(sft_results)}, File saved to {sft_out_path}")

    dpo_out_path = os.path.join(base_dir, "MMPR-v1.2", f"dpo_mmpr_{str(DATA_SIZE)}_{SEED}.json")
    save_json(dpo_results, dpo_out_path)
    print(f"Total DPO samples: {len(dpo_results)}, File saved to {dpo_out_path}")


if __name__ == "__main__":
    main()