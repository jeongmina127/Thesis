from pathlib import Path
import shutil
import random

SRC = Path(
    "/mnt/c/Users/Jeongmin Cho/Desktop/Git Repository/Thesis/"
    "WiSARD_Multi_Modal_Sample/VIS_tiled"
)

DST = Path(
    "/mnt/c/Users/Jeongmin Cho/Desktop/Git Repository/Thesis/"
    "WiSARD_Multi_Modal_Sample/VIS_culled"
)

splits = ["train", "valid", "test"]

NEGATIVE_RATIO = 0.001   # 0.1%
SEED = 42

random.seed(SEED)


def find_image(image_dir, stem):
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def copy_pair(label_path, image_path, dst_labels, dst_images):
    shutil.copy2(
        label_path,
        dst_labels / label_path.name
    )

    shutil.copy2(
        image_path,
        dst_images / image_path.name
    )


for split in splits:

    src_labels = SRC / split / "labels"
    src_images = SRC / split / "images"

    dst_labels = DST / split / "labels"
    dst_images = DST / split / "images"

    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    positive_labels = []
    negative_labels = []

    # --------------------------------------------------
    # 1. positive / negative 구분
    # --------------------------------------------------
    for label_path in src_labels.glob("*.txt"):

        with open(label_path, "r", encoding="utf-8") as f:
            has_human = bool(f.read().strip())

        if has_human:
            positive_labels.append(label_path)
        else:
            negative_labels.append(label_path)

    # --------------------------------------------------
    # 2. negative 중 0.1%만 랜덤 선택
    # --------------------------------------------------
    n_negative_keep = int(len(negative_labels) * NEGATIVE_RATIO)

    selected_negative_labels = random.sample(
        negative_labels,
        n_negative_keep
    )

    # 최종적으로 사용할 label
    selected_labels = positive_labels + selected_negative_labels

    # --------------------------------------------------
    # 3. image + label 복사
    # --------------------------------------------------
    for label_path in selected_labels:

        image_path = find_image(
            src_images,
            label_path.stem
        )

        if image_path is None:
            print(f"WARNING: image not found for {label_path.name}")
            continue

        copy_pair(
            label_path,
            image_path,
            dst_labels,
            dst_images
        )

    # --------------------------------------------------
    # 결과 출력
    # --------------------------------------------------
    print(f"\n[{split}]")
    print(f"Positive tiles       : {len(positive_labels)}")
    print(f"Negative tiles total : {len(negative_labels)}")
    print(f"Negative tiles kept  : {len(selected_negative_labels)}")
    print(f"Final dataset        : {len(selected_labels)}")