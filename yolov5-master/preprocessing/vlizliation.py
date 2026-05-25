import os
import cv2
import numpy as np

# ============================================================
# Visualize YOLOv5 Segmentation TXT Labels
# + Auto Save Visualized Images
# + Auto Create Output Folder
# ============================================================

# Image folder
IMAGE_FOLDER = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\Client_annotated_images\images"

# Label folder
LABEL_FOLDER = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\Client_annotated_images\labels_yolo"
# Output folder
OUTPUT_FOLDER = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\Client_annotated_images\visualized_output"

# Create output folder automatically
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Class colors
CLASS_COLORS = {
    0: (0, 255, 0),     # Green
    1: (255, 0, 0),     # Blue
    2: (0, 0, 255),     # Red
}

# ============================================================

for image_name in os.listdir(IMAGE_FOLDER):

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    ext = os.path.splitext(image_name)[1].lower()

    if ext not in IMAGE_EXTENSIONS:
        continue

    # Corresponding txt path
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(LABEL_FOLDER, txt_name)

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read: {image_path}")
        continue

    h, w = image.shape[:2]

    # Skip if no label exists
    if not os.path.exists(txt_path):
        print(f"No label found for {image_name}")
        continue

    # Read label file
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        data = line.strip().split()

        # Minimum polygon points check
        if len(data) < 7:
            continue

        class_id = int(data[0])

        coords = list(map(float, data[1:]))

        points = []

        # Convert normalized coords -> pixel coords
        for i in range(0, len(coords), 2):

            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)

            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # Get color
        color = CLASS_COLORS.get(class_id, (255, 255, 0))

        # Draw polygon border
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=color,
            thickness=2
        )

        # Create overlay for transparency
        overlay = image.copy()

        # Fill polygon
        cv2.fillPoly(
            overlay,
            [points],
            color
        )

        # Transparency
        alpha = 0.35

        image = cv2.addWeighted(
            overlay,
            alpha,
            image,
            1 - alpha,
            0
        )

        # Add label text
        x_text, y_text = points[0]

        cv2.putText(
            image,
            f"Class {class_id}",
            (x_text, y_text - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    save_path = os.path.join(OUTPUT_FOLDER, image_name)

    cv2.imwrite(save_path, image)

    print(f"Saved: {save_path}")

    # ========================================================
    # OPTIONAL DISPLAY
    # ========================================================

    cv2.imshow("YOLO Segmentation Viewer", image)

    key = cv2.waitKey(1)

    # Press ESC to stop
    if key == 27:
        break

cv2.destroyAllWindows()

print("\nAll visualized images saved successfully!")