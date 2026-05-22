import os
import json
from glob import glob

# ============================================================
# Convert LabelMe JSON Polygon Annotations to YOLOv5 Seg TXT
# ============================================================

# Folder containing JSON files
JSON_FOLDER = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\dataset_new\labels"

# Output folder for YOLO segmentation labels
OUTPUT_FOLDER = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\dataset_new\labels_yolo"  # Change as needed
# Class mapping
# Example:
# "person": 0,
# "car": 1
CLASS_MAP = {
    "sack": 0
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all JSON files
json_files = glob(os.path.join(JSON_FOLDER, "*.json"))

for json_file in json_files:

    with open(json_file, "r") as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    output_lines = []

    for shape in data["shapes"]:

        label = shape["label"]

        # Skip labels not in CLASS_MAP
        if label not in CLASS_MAP:
            print(f"Skipping unknown class: {label}")
            continue

        class_id = CLASS_MAP[label]

        points = shape["points"]

        # YOLO segmentation format:
        # class_id x1 y1 x2 y2 x3 y3 ...
        seg_points = []

        for point in points:
            x = point[0] / image_width
            y = point[1] / image_height

            # Clamp values between 0 and 1
            x = min(max(x, 0), 1)
            y = min(max(y, 0), 1)

            seg_points.extend([f"{x:.6f}", f"{y:.6f}"])

        line = f"{class_id} " + " ".join(seg_points)
        output_lines.append(line)

    # Save TXT file
    txt_filename = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
    txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)

    with open(txt_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Converted: {json_file} -> {txt_path}")

print("\nAll JSON files converted successfully!")