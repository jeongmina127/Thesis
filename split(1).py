from pathlib import Path
import random
import shutil

# ============================================================
# SETTINGS
# ============================================================

SRC = Path(
    "/mnt/c/Users/Jeongmin Cho/Desktop/Git Repository/Thesis/"
    "WiSARD_Multi_Modal_Sample/210417_MtErie_Enterprise_VIS_0003"
)

# Current unsplit data
IMAGE_DIR = SRC / "train" / "images"
LABEL_DIR = SRC / "train" / "labels"

# Write split dataset somewhere new
DST = Path(
    "/mnt/c/Users/Jeongmin Cho/Desktop/Git Repository/Thesis/"
    "WiSARD_Multi_Modal_Sample/210417_MtErie_Enterprise_VIS_0003_split"
)

TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
TEST_RATIO = 0.20

SEED = 42

# Copy files rather than move them
COPY_FILES = True


# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) < 1e-8


# ============================================================
# FIND IMAGE-LABEL PAIRS
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
)

pairs = []

for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"WARNING: no label for {image_path.name}")
        continue

    pairs.append((image_path, label_path))


print(f"Found {len(pairs)} image-label pairs.")


# ============================================================
# SHUFFLE
# ============================================================

random.seed(SEED)
random.shuffle(pairs)


# ============================================================
# SPLIT
# ============================================================

n_total = len(pairs)

n_train = int(n_total * TRAIN_RATIO)
n_valid = int(n_total * VALID_RATIO)

train_pairs = pairs[:n_train]
valid_pairs = pairs[n_train:n_train + n_valid]
test_pairs = pairs[n_train + n_valid:]


print(f"Train : {len(train_pairs)}")
print(f"Valid : {len(valid_pairs)}")
print(f"Test  : {len(test_pairs)}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "valid", "test"]:

    (DST / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (DST / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY / MOVE
# ============================================================

def save_split(pairs, split_name):

    image_dst = DST / split_name / "images"
    label_dst = DST / split_name / "labels"

    for image_path, label_path in pairs:

        if COPY_FILES:

            shutil.copy2(
                image_path,
                image_dst / image_path.name
            )

            shutil.copy2(
                label_path,
                label_dst / label_path.name
            )

        else:

            shutil.move(
                str(image_path),
                image_dst / image_path.name
            )

            shutil.move(
                str(label_path),
                label_dst / label_path.name
            )


save_split(train_pairs, "train")
save_split(valid_pairs, "valid")
save_split(test_pairs, "test")


print("\nDone.")
print(f"Output: {DST}")