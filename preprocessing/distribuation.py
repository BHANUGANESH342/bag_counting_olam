import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def distribute_images_equally(
    source_folder,
    output_folder,
    people_list,
    image_extensions=(".jpg", ".jpeg", ".png", ".bmp")
):
    """
    Shuffle and distribute images equally into
    separate folders for each person.

    Output Structure:
    output_folder/
        bhanu/
        arya/
        rahul/
        sai/
    """

    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    # ==========================================
    # Create Main Output Folder
    # ==========================================
    output_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Get All Images
    # ==========================================
    images = []

    for ext in image_extensions:
        images.extend(source_folder.glob(f"*{ext}"))

    if len(images) == 0:
        print("No images found")
        return

    print(f"Total Images Found : {len(images)}")

    # ==========================================
    # Shuffle Images
    # ==========================================
    random.shuffle(images)

    # ==========================================
    # Create Separate Folder for Each Person
    # ==========================================
    for person in people_list:

        person_folder = output_folder / person

        person_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Distribute Images
    # ==========================================
    distribution_count = defaultdict(int)

    pbar = tqdm(images, desc="Distributing Images")

    for idx, image_path in enumerate(pbar):

        # Assign equally
        person = people_list[idx % len(people_list)]

        person_folder = output_folder / person

        # Rename image to avoid duplicate names
        new_image_name = f"{person}_{distribution_count[person]:06d}{image_path.suffix}"

        destination = person_folder / new_image_name

        shutil.copy2(image_path, destination)

        distribution_count[person] += 1

    # ==========================================
    # Summary
    # ==========================================
    print("\nDistribution Completed\n")

    for person in people_list:

        print(f"{person:<15} : {distribution_count[person]} images")


# ==========================================
# Example Usage
# ==========================================

source_folder = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\client_video\8_fps_frames"

output_folder = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\client_video"

people_list = [
    "bindu",
    "pojitha",
    "mahesh",
    "varun",
    "jk",
    "rohit",
    "anil",
    "gireesh",
    "sai krishna",
]

distribute_images_equally(
    source_folder,
    output_folder,
    people_list
)