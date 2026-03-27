import os
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    opt = parser.parse_args()

    data_root = os.path.abspath(opt.data_root)
    save_root = os.path.join(data_root, "texture_dataset")
    os.makedirs(save_root, exist_ok=True)

    src_transforms = os.path.join(data_root, "mesh", "transforms.json")
    if not os.path.isfile(src_transforms):
        raise FileNotFoundError(f"Source transforms not found: {src_transforms}")

    image_root = os.path.join(save_root, "image")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Selected image directory not found: {image_root}")

    selected_meta_path = os.path.join(save_root, "select_sharp.json")
    if os.path.isfile(selected_meta_path):
        with open(selected_meta_path, "r") as f:
            meta = json.load(f)
        filtered_frames = meta.get("frames", [])
    else:
        with open(src_transforms, "r") as f:
            meta = json.load(f)

        img_files = {
            name
            for name in os.listdir(image_root)
            if os.path.isfile(os.path.join(image_root, name))
        }
        filtered_frames = [f for f in meta.get("frames", []) if os.path.basename(f["file_path"]) in img_files]
    if not filtered_frames:
        raise RuntimeError("No frames left after filtering by selected images.")

    meta["frames"] = filtered_frames
    with open(os.path.join(save_root, "transforms.json"), "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()
