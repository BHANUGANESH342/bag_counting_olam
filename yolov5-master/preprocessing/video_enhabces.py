# import cv2
# import numpy as np
# import shutil
# from pathlib import Path
# from tqdm import tqdm
# from datetime import datetime


# # =====================================================
# # SUPER RESOLUTION + AUTO FOLDER CREATION
# # =====================================================
# def super_resolution_video(
#     input_video,
#     output_base_folder,
#     scale=1.5
# ):
#     """
#     Features:
#     --------------------------------
#     1. Auto creates output folders
#     2. Keeps original video backup
#     3. Enhances blurry CCTV footage
#     4. Upscales resolution
#     5. Sharpens frames
#     6. Contrast enhancement
#     """

#     # =====================================================
#     # INPUT PATH
#     # =====================================================
#     input_video = Path(input_video)

#     if not input_video.exists():
#         print("Input video not found")
#         return

#     # =====================================================
#     # CREATE MAIN OUTPUT FOLDER
#     # =====================================================
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     video_name = input_video.stem

#     main_output_folder = Path(output_base_folder) / f"{video_name}_{timestamp}"

#     main_output_folder.mkdir(parents=True, exist_ok=True)

#     # =====================================================
#     # CREATE SUBFOLDERS
#     # =====================================================
#     original_folder = main_output_folder / "original_video"

#     enhanced_folder = main_output_folder / "enhanced_video"

#     original_folder.mkdir(exist_ok=True)

#     enhanced_folder.mkdir(exist_ok=True)

#     # =====================================================
#     # COPY ORIGINAL VIDEO
#     # =====================================================
#     original_copy_path = original_folder / input_video.name

#     shutil.copy2(input_video, original_copy_path)

#     print(f"\nOriginal video copied to:")
#     print(original_copy_path)

#     # =====================================================
#     # OUTPUT ENHANCED VIDEO PATH
#     # =====================================================
#     output_video = enhanced_folder / f"{video_name}_super_resolution.mp4"

#     # =====================================================
#     # OPEN VIDEO
#     # =====================================================
#     cap = cv2.VideoCapture(str(input_video))

#     if not cap.isOpened():
#         print("Error opening video")
#         return

#     # =====================================================
#     # VIDEO PROPERTIES
#     # =====================================================
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fps = cap.get(cv2.CAP_PROP_FPS)

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     print(f"\nOriginal Resolution : {width}x{height}")

#     print(f"FPS                 : {fps}")

#     print(f"Total Frames        : {total_frames}")

#     # =====================================================
#     # UPSCALED SIZE
#     # =====================================================
#     new_width = width * scale

#     new_height = height * scale

#     print(f"Upscaled Resolution : {new_width}x{new_height}")

#     # =====================================================
#     # VIDEO WRITER
#     # =====================================================
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     out = cv2.VideoWriter(
#         str(output_video),
#         fourcc,
#         fps,
#         (new_width, new_height)
#     )

#     # =====================================================
#     # PROGRESS BAR
#     # =====================================================
#     pbar = tqdm(
#         total=total_frames,
#         desc="Enhancing Video"
#     )

#     # =====================================================
#     # PROCESS VIDEO
#     # =====================================================
#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         # =================================================
#         # 1. DENOISE
#         # =================================================
#         frame = cv2.fastNlMeansDenoisingColored(
#             frame,
#             None,
#             5,
#             5,
#             7,
#             21
#         )

#         # =================================================
#         # 2. UPSCALE
#         # =================================================
#         upscaled = cv2.resize(
#             frame,
#             (new_width, new_height),
#             interpolation=cv2.INTER_LANCZOS4
#         )

#         # =================================================
#         # 3. SHARPEN
#         # =================================================
#         sharpen_kernel = np.array([
#             [-1, -1, -1],
#             [-1,  9, -1],
#             [-1, -1, -1]
#         ])

#         sharpened = cv2.filter2D(
#             upscaled,
#             -1,
#             sharpen_kernel
#         )

#         # =================================================
#         # 4. CONTRAST ENHANCEMENT
#         # =================================================
#         lab = cv2.cvtColor(
#             sharpened,
#             cv2.COLOR_BGR2LAB
#         )

#         l, a, b = cv2.split(lab)

#         clahe = cv2.createCLAHE(
#             clipLimit=3.0,
#             tileGridSize=(8, 8)
#         )

#         cl = clahe.apply(l)

#         enhanced_lab = cv2.merge((cl, a, b))

#         enhanced = cv2.cvtColor(
#             enhanced_lab,
#             cv2.COLOR_LAB2BGR
#         )

#         # =================================================
#         # 5. BRIGHTNESS IMPROVEMENT
#         # =================================================
#         enhanced = cv2.convertScaleAbs(
#             enhanced,
#             alpha=1.1,
#             beta=5
#         )

#         # =================================================
#         # WRITE FRAME
#         # =================================================
#         out.write(enhanced)

#         pbar.update(1)

#     # =====================================================
#     # CLEANUP
#     # =====================================================
#     pbar.close()

#     cap.release()

#     out.release()

#     # =====================================================
#     # SUMMARY
#     # =====================================================
#     print("\n====================================")
#     print("PROCESS COMPLETED")
#     print("====================================")

#     print(f"\nMain Folder:")
#     print(main_output_folder)

#     print(f"\nOriginal Video:")
#     print(original_copy_path)

#     print(f"\nEnhanced Video:")
#     print(output_video)


# # =====================================================
# # EXAMPLE USAGE
# # =====================================================

# input_video = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\input_videos\olam_client_video_v2.mp4"
# # USER GIVEN OUTPUT ADDRESS
# output_base_folder = r"D:\bhanu\olam_agri\client_video"

# super_resolution_video(
#     input_video=input_video,
#     output_base_folder=output_base_folder,
#     scale=2
# )





























import cv2
import numpy as np
import shutil
import torch
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from datetime import datetime


# =====================================================
# CHECK GPU
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device : {device}")

if device == "cuda":
    print(torch.cuda.get_device_name(0))


# =====================================================
# PYTORCH GPU UPSCALING
# =====================================================
def upscale_frame_gpu(frame, target_size):

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(frame_rgb).float()

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    tensor = tensor / 255.0

    # GPU Upscaling
    upscaled = F.interpolate(
        tensor,
        size=target_size,
        mode="bicubic",
        align_corners=False
    )

    upscaled = upscaled.squeeze(0)

    upscaled = upscaled.permute(1, 2, 0)

    upscaled = (
        upscaled.clamp(0, 1)
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)

    upscaled = cv2.cvtColor(
        upscaled,
        cv2.COLOR_RGB2BGR
    )

    return upscaled


# =====================================================
# MAIN ENHANCEMENT FUNCTION
# =====================================================
def enhance_to_4k_gpu(
    input_video,
    output_base_folder
):

    input_video = Path(input_video)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_name = input_video.stem

    main_output_folder = (
        Path(output_base_folder)
        / f"{video_name}_{timestamp}"
    )

    main_output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # =================================================
    # FOLDERS
    # =================================================
    original_folder = (
        main_output_folder
        / "original_video"
    )

    enhanced_folder = (
        main_output_folder
        / "enhanced_4k_gpu"
    )

    original_folder.mkdir(exist_ok=True)

    enhanced_folder.mkdir(exist_ok=True)

    # =================================================
    # COPY ORIGINAL
    # =================================================
    shutil.copy2(
        input_video,
        original_folder / input_video.name
    )

    # =================================================
    # OUTPUT VIDEO
    # =================================================
    output_video = (
        enhanced_folder
        / f"{video_name}_4k_gpu.mp4"
    )

    # =================================================
    # OPEN VIDEO
    # =================================================
    cap = cv2.VideoCapture(str(input_video))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"\nOriginal Resolution : {width}x{height}")

    print(f"FPS                 : {fps}")

    print(f"Total Frames        : {total_frames}")

    # =================================================
    # 4K SIZE
    # =================================================
    target_width = 3840

    target_height = 2160

    print(f"\nOutput Resolution   : {target_width}x{target_height}")

    # =================================================
    # VIDEO WRITER
    # =================================================
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (target_width, target_height)
    )

    # =================================================
    # SHARPEN KERNEL
    # =================================================
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # =================================================
    # PROGRESS BAR
    # =================================================
    pbar = tqdm(
        total=total_frames,
        desc="GPU 4K Enhancement"
    )

    # =================================================
    # PROCESS VIDEO
    # =================================================
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # =============================================
        # DENOISE
        # =============================================
        denoise = cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            8,
            8,
            7,
            21
        )

        # =============================================
        # DEBLUR
        # =============================================
        blur = cv2.GaussianBlur(
            denoise,
            (0, 0),
            3
        )

        deblur = cv2.addWeighted(
            denoise,
            2.0,
            blur,
            -1.0,
            0
        )

        # =============================================
        # GPU UPSCALING
        # =============================================
        upscaled = upscale_frame_gpu(
            deblur,
            (target_height, target_width)
        )

        # =============================================
        # SHARPEN
        # =============================================
        sharpened = cv2.filter2D(
            upscaled,
            -1,
            sharpen_kernel
        )

        # =============================================
        # CLAHE
        # =============================================
        lab = cv2.cvtColor(
            sharpened,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)

        enhanced_lab = cv2.merge((cl, a, b))

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        # =============================================
        # BRIGHTNESS
        # =============================================
        enhanced = cv2.convertScaleAbs(
            enhanced,
            alpha=1.15,
            beta=8
        )

        # =============================================
        # WRITE FRAME
        # =============================================
        out.write(enhanced)

        pbar.update(1)

    # =================================================
    # CLEANUP
    # =================================================
    pbar.close()

    cap.release()

    out.release()

    print("\n===================================")
    print("GPU 4K ENHANCEMENT COMPLETED")
    print("===================================")

    print(f"\nSaved To:")
    print(output_video)


# =====================================================
# EXAMPLE USAGE
# =====================================================
input_video = (
    r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\input_videos\olam_client_video_v2.mp4"
)
output_base_folder = (
    r"D:\bhanu\olam_agri\client_video"
)

enhance_to_4k_gpu(
    input_video=input_video,
    output_base_folder=output_base_folder
)