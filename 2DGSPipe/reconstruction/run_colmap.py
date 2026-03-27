import argparse
import os
import subprocess
from typing import Iterable


def run_cmd(cmd: Iterable[str]) -> None:
    subprocess.run(list(cmd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    data_root = os.path.abspath(args.data_root)

    img_root = os.path.join(data_root, "images")
    mask_root = os.path.join(data_root, "wholebody_mask")
    database_root = os.path.join(data_root, "database.db")
    sparse_root = os.path.join(data_root, "sparse")
    recon_root = os.path.join(sparse_root, "0")
    camera_model = "PINHOLE"

    run_cmd(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            database_root,
            "--image_path",
            img_root,
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.max_image_size",
            "4000",
            "--FeatureExtraction.use_gpu",
            "1",
            "--ImageReader.mask_path",
            mask_root,
        ]
    )

    run_cmd(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            database_root,
            "--FeatureMatching.guided_matching",
            "1",
            "--FeatureMatching.use_gpu",
            "1",
        ]
    )

    os.makedirs(sparse_root, exist_ok=True)
    run_cmd(
        [
            "colmap",
            "mapper",
            "--database_path",
            database_root,
            "--image_path",
            img_root,
            "--output_path",
            sparse_root,
        ]
    )
    run_cmd(
        [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            recon_root,
            "--output_path",
            recon_root,
            "--BundleAdjustment.max_num_iterations",
            "100",
        ]
    )
    run_cmd(
        [
            "colmap",
            "model_converter",
            "--input_path",
            recon_root,
            "--output_path",
            recon_root,
            "--output_type",
            "TXT",
        ]
    )


if __name__ == "__main__":
    main()
