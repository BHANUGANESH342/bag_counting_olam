# import os
# import shutil
# import random
# from tqdm import tqdm
 
# """
# ========================================================
# Dataset Split Script (Train / Validation)
# ========================================================
 
# Purpose:
# --------
# This script splits an image dataset and its corresponding YOLO label files
# into training and validation (test) sets.
 
# It:
# - Reads images from `images_` folder
# - Reads labels from `labels_` folder
# - Randomly shuffles the dataset
# - Splits data into train and valid sets
# - Moves files into YOLO-compatible directory structure
# - Warns if any image does not have a corresponding label
 
# Typical use case:
# -----------------
# Preparing datasets for YOLOv5 / YOLOv8 / YOLOv9 / YOLOv11 training.
# """
 
# # =========================
# # Base paths (Root dataset directory)
# # =========================
# BASE_PATH = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\dataset_new"
 
# # Source directories (original data)
# IMAGES_DIR = os.path.join(BASE_PATH, "images")
# LABELS_DIR = os.path.join(BASE_PATH, "labels_yolo")
 
# # Destination directories (after split)
# TRAIN_IMG = os.path.join(BASE_PATH, "train", "images")
# TRAIN_LBL = os.path.join(BASE_PATH, "train", "labels")
# TEST_IMG = os.path.join(BASE_PATH, "valid", "images")
# TEST_LBL = os.path.join(BASE_PATH, "valid", "labels")
 
# # =========================
# # Create output folders if they do not exist
# # =========================
# os.makedirs(TRAIN_IMG, exist_ok=True)
# os.makedirs(TRAIN_LBL, exist_ok=True)
# os.makedirs(TEST_IMG, exist_ok=True)
# os.makedirs(TEST_LBL, exist_ok=True)
 
# # =========================
# # Collect all image files
# # =========================
# # Reads all images with valid extensions from the images folder
# image_files = [
#     f for f in os.listdir(IMAGES_DIR)
#     if f.lower().endswith((".jpg", ".png", ".jpeg"))
# ]
 
# # Shuffle images to ensure random distribution
# random.shuffle(image_files)
 
# # =========================
# # Split configuration
# # =========================
# test_split = 0.2  # 20% of data will go to validation set
# num_test = int(len(image_files) * test_split)
 
# # Split file lists
# test_files = image_files[:num_test]
# train_files = image_files[num_test:]
 
# # Display dataset statistics
# print(f"Total images : {len(image_files)}")
# print(f"Train images : {len(train_files)}")
# print(f"Test images  : {len(test_files)}")
 
# # =========================
# # Move TRAIN files
# # =========================
# for img_file in tqdm(train_files, desc="Moving TRAIN files"):
#     # Create label filename from image filename
#     lbl_file = os.path.splitext(img_file)[0] + ".txt"
 
#     # Move image to train/images
#     shutil.move(
#         os.path.join(IMAGES_DIR, img_file),
#         os.path.join(TRAIN_IMG, img_file)
#     )
 
#     # Move label to train/labels if it exists
#     lbl_src = os.path.join(LABELS_DIR, lbl_file)
#     if os.path.exists(lbl_src):
#         shutil.move(lbl_src, os.path.join(TRAIN_LBL, lbl_file))
#     else:
#         print(f"⚠️ Missing label for {img_file}")
 
# # =========================
# # Move TEST files
# # =========================
# for img_file in tqdm(test_files, desc="Moving TEST files"):
#     # Create label filename from image filename
#     lbl_file = os.path.splitext(img_file)[0] + ".txt"
 
#     # Move image to valid/images
#     shutil.move(
#         os.path.join(IMAGES_DIR, img_file),
#         os.path.join(TEST_IMG, img_file)
#     )
 
#     # Move label to valid/labels if it exists
#     lbl_src = os.path.join(LABELS_DIR, lbl_file)
#     if os.path.exists(lbl_src):
#         shutil.move(lbl_src, os.path.join(TEST_LBL, lbl_file))
#     else:
#         print(f"⚠️ Missing label for {img_file}")
 
# # =========================
# # Completion message
# # =========================
# print("✅ Dataset split completed successfully!")
 

























import os
import shutil
import random
from tqdm import tqdm

"""
========================================================
Dataset Split Script (Train / Validation)
COPY VERSION (Original data remains untouched)
========================================================
"""

# =========================
# Base paths
# =========================
BASE_PATH = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\Client_annotated_images"

# Source directories
IMAGES_DIR = os.path.join(BASE_PATH, "images")
LABELS_DIR = os.path.join(BASE_PATH, "labels")

# Destination directories
TRAIN_IMG = os.path.join(BASE_PATH, "train", "images")
TRAIN_LBL = os.path.join(BASE_PATH, "train", "labels")

VALID_IMG = os.path.join(BASE_PATH, "valid", "images")
VALID_LBL = os.path.join(BASE_PATH, "valid", "labels")

# =========================
# Create folders automatically
# =========================
os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)

os.makedirs(VALID_IMG, exist_ok=True)
os.makedirs(VALID_LBL, exist_ok=True)

# =========================
# Read image files
# =========================
image_files = [
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Shuffle dataset
random.shuffle(image_files)

# =========================
# Split ratio
# =========================
valid_split = 0.2

num_valid = int(len(image_files) * valid_split)

valid_files = image_files[:num_valid]
train_files = image_files[num_valid:]

# =========================
# Dataset statistics
# =========================
print(f"\nTotal Images : {len(image_files)}")
print(f"Train Images : {len(train_files)}")
print(f"Valid Images : {len(valid_files)}\n")

# ============================================================
# COPY TRAIN FILES
# ============================================================
for img_file in tqdm(train_files, desc="Copying TRAIN files"):

    lbl_file = os.path.splitext(img_file)[0] + ".txt"

    src_img = os.path.join(IMAGES_DIR, img_file)
    dst_img = os.path.join(TRAIN_IMG, img_file)

    # Copy image
    shutil.copy2(src_img, dst_img)

    # Copy label
    src_lbl = os.path.join(LABELS_DIR, lbl_file)
    dst_lbl = os.path.join(TRAIN_LBL, lbl_file)

    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        print(f"⚠️ Missing label for: {img_file}")

# ============================================================
# COPY VALID FILES
# ============================================================
for img_file in tqdm(valid_files, desc="Copying VALID files"):

    lbl_file = os.path.splitext(img_file)[0] + ".txt"

    src_img = os.path.join(IMAGES_DIR, img_file)
    dst_img = os.path.join(VALID_IMG, img_file)

    # Copy image
    shutil.copy2(src_img, dst_img)

    # Copy label
    src_lbl = os.path.join(LABELS_DIR, lbl_file)
    dst_lbl = os.path.join(VALID_LBL, lbl_file)

    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        print(f"⚠️ Missing label for: {img_file}")

# =========================
# Completion Message
# =========================
print("\n✅ Dataset copied and split completed successfully!")
print("Original dataset remains unchanged.")