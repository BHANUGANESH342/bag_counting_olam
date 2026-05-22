import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames_at_target_fps(video_path, target_fps):
    """
    Extract frames from video at target FPS with progress bar.
    """

    # Convert video path to Path object
    video_path = Path(video_path)

    # ==========================================
    # Auto Create Output Folder
    # ==========================================
    output_folder = Path(
        r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\data_set_video_3_\client_video\8_fps_frames"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Open Video
    # ==========================================
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Unable to open video")
        return

    # ==========================================
    # Video Properties
    # ==========================================
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original FPS : {original_fps}")
    print(f"Target FPS   : {target_fps}")
    print(f"Total Frames : {total_frames}")

    # ==========================================
    # Safety Check
    # ==========================================
    if target_fps > original_fps:
        print("Target FPS cannot be greater than original FPS")
        cap.release()
        return

    # ==========================================
    # Calculate Frame Interval
    # ==========================================
    frame_interval = int(original_fps / target_fps)

    frame_count = 0
    saved_count = 0

    # ==========================================
    # Progress Bar
    # ==========================================
    pbar = tqdm(total=total_frames, desc="Processing Video")

    # ==========================================
    # Read Video Frames
    # ==========================================
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Save only required frames
        if frame_count % frame_interval == 0:

            frame_name = output_folder / f"frame_{saved_count:06d}.jpg"

            cv2.imwrite(str(frame_name), frame)

            saved_count += 1

        frame_count += 1

        # Update Progress Bar
        pbar.update(1)

    # ==========================================
    # Close Progress Bar
    # ==========================================
    pbar.close()

    # ==========================================
    # Release Video
    # ==========================================
    cap.release()

    print("\nProcessing Completed")
    print(f"Total Frames Read : {frame_count}")
    print(f"Frames Saved      : {saved_count}")
    print(f"Output Folder     : {output_folder}")


# ==========================================
# Example Usage
# ==========================================

video_path = r"C:\Users\admin\Downloads\olam_client_video_v2.mp4"

target_fps = 8

extract_frames_at_target_fps(
    video_path,
    target_fps
)