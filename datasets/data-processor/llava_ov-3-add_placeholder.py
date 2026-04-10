import os
import json
import yaml
from tqdm import tqdm

def process_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Succesfully open {os.path.basename(path)}")

    added = 0
    dup_cleaned = 0

    for item in tqdm(data, desc=f"Processing {os.path.basename(path)}"):
        convs = item.get('conversations')
        # only process items that have an image field and a conversations list
        if 'image' in item and isinstance(convs, list):
            # find all human conversation indices
            human_idxs = [i for i, c in enumerate(convs) if c.get('from') == 'human']
            if not human_idxs:
                continue

            # count total '<image>' across all human convs
            total_imgs = sum(
                convs[i].get('value', '').count('<image>')
                for i in human_idxs
            )

            if total_imgs == 0:
                # no placeholder at all: add to first human conv
                idx = human_idxs[0]
                convs[idx]['value'] = '<image>\n' + convs[idx].get('value', '')
                added += 1

            elif total_imgs > 1:
                # more than one placeholder: remove all, then add one at first
                for i in human_idxs:
                    cleaned = convs[i]['value'].replace('<image>', '').strip()
                    convs[i]['value'] = cleaned
                first = human_idxs[0]
                convs[first]['value'] = '<image>\n' + convs[first]['value']
                dup_cleaned += 1

    # overwrite only if we made changes
    if added or dup_cleaned:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{os.path.basename(path)}: {added} placeholders added, {dup_cleaned} items cleaned")


def main():
    # 1) load YAML
    single_image_yaml = os.path.join(os.path.dirname(__file__), "OneVisionData", "single_image.yaml")
    with open(single_image_yaml, 'r', encoding='utf-8') as yf:
        cfg = yaml.safe_load(yf)

    # 2) iterate over datasets
    for ds in cfg.get('datasets', []):
        jp = ds.get('json_path')
        if jp and os.path.isfile(jp):
            process_json(jp)


if __name__ == '__main__':
    main()
