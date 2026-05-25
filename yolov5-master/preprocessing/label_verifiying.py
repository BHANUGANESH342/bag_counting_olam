import shutil
from pathlib import Path
from tqdm import tqdm


def check_images_and_labels(
    image_folder,
    label_folder,
    output_folder,
    copy_files=True
):
    """
    Verify image-label pairs.

    Output Structure
    --------------------------------
    output_folder/
        matched/
            images/
            labels/

        missing_labels/
            images/

        orphan_labels/
            labels/
    """

    # ==========================================
    # Convert to Path Objects
    # ==========================================
    image_folder = Path(image_folder)

    label_folder = Path(label_folder)

    output_folder = Path(output_folder)

    # ==========================================
    # Create Output Directories
    # ==========================================
    matched_images = output_folder / "matched" / "images"

    matched_labels = output_folder / "matched" / "labels"

    missing_images = output_folder / "missing_labels" / "images"

    orphan_labels = output_folder / "orphan_labels" / "labels"

    matched_images.mkdir(parents=True, exist_ok=True)

    matched_labels.mkdir(parents=True, exist_ok=True)

    missing_images.mkdir(parents=True, exist_ok=True)

    orphan_labels.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Get Image Files
    # ==========================================
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP"
    }

    image_files = [
        f for f in image_folder.iterdir()
        if f.suffix in image_extensions
    ]

    label_files = list(label_folder.glob("*.txt"))

    # ==========================================
    # Create Name Sets
    # ==========================================
    image_names = {img.stem for img in image_files}

    label_names = {lbl.stem for lbl in label_files}

    print(f"\nTotal Images : {len(image_files)}")

    print(f"Total Labels : {len(label_files)}")

    # ==========================================
    # Copy or Move Function
    # ==========================================
    operation = shutil.copy2 if copy_files else shutil.move

    # ==========================================
    # CHECK IMAGE-LABEL MATCHES
    # ==========================================
    matched_count = 0

    missing_count = 0

    print("\nChecking Image-Label Pairs...\n")

    for image_path in tqdm(image_files):

        image_name = image_path.stem

        label_path = label_folder / f"{image_name}.txt"

        # ======================================
        # MATCH FOUND
        # ======================================
        if label_path.exists():

            operation(
                image_path,
                matched_images / image_path.name
            )

            operation(
                label_path,
                matched_labels / label_path.name
            )

            matched_count += 1

        # ======================================
        # MISSING LABEL
        # ======================================
        else:

            operation(
                image_path,
                missing_images / image_path.name
            )

            missing_count += 1

    # ==========================================
    # CHECK ORPHAN LABELS
    # ==========================================
    orphan_count = 0

    print("\nChecking Orphan Labels...\n")

    for label_path in tqdm(label_files):

        if label_path.stem not in image_names:

            operation(
                label_path,
                orphan_labels / label_path.name
            )

            orphan_count += 1

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n===================================")

    print("VERIFICATION COMPLETED")

    print("===================================")

    print(f"\nMatched Pairs   : {matched_count}")

    print(f"Missing Labels  : {missing_count}")

    print(f"Orphan Labels   : {orphan_count}")

    print(f"\nOutput Folder:")

    print(output_folder)


# ==========================================
# EXAMPLE USAGE
# ==========================================

image_folder = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\Client_annotated_images\images"

label_folder = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\Client_annotated_images\labels"

output_folder = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\Client_annotated_images\verification_output"

check_images_and_labels(
    image_folder=image_folder,
    label_folder=label_folder,
    output_folder=output_folder,
    copy_files=True
)