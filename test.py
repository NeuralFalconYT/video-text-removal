import os
import sys
path = "."

if path not in sys.path:
    sys.path.insert(0, path)

from mi_gan.image_inpainting import MIGAN
from mi_gan.schema import InpaintRequest
from ocr.utils import run_ocr,draw_polygons,generate_text_mask,blur_text_regions
import torch
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MIGAN(device=device)
config = InpaintRequest(
    hd_strategy="crop",
    hd_strategy_crop_margin=128,
)






def local_ocr(image,bounding_box=True,hide_method=0):
  result=image
  result_path=None
  if isinstance(image, np.ndarray):
    frame_path="./frame.png"
    img = image
  if isinstance(image, str):
    frame_path=image
    img = cv2.imread(image)
  raw_img=img.copy()
  ocr_items=run_ocr(img)
  if bounding_box:
    raw_img=draw_polygons(raw_img, ocr_items, display_text=False)
  if hide_method==1:
    method="inpaint"
  else:
    method="blur"
  if method=="blur":
    blurred_img = blur_text_regions(img,
                                  ocr_items,
                                  blur_level=3,
                                  blur_sigma=0,
                                  min_conf=0.0,
                                  expand_scale=1.0)
    result=blurred_img
    base, ext = os.path.splitext(frame_path)
    blur_image_path = f"{base}_blur{ext}"
    cv2.imwrite(blur_image_path, blurred_img)
    result_path=blur_image_path
  if method=="inpaint":
    mask =  generate_text_mask(img,
                              ocr_items,
                              invert=False,
                              min_conf=0.0,
                              expand_scale=1.0
                            )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_inpaint = model(img_rgb, mask, config)
    result=output_inpaint
    base, ext = os.path.splitext(frame_path)
    mask_path = f"{base}_inpaint{ext}"
    cv2.imwrite(mask_path, output_inpaint)
    result_path=mask_path
  return raw_img,result,result_path



import os
import cv2
from tqdm.auto import tqdm   


def extract_and_write_chunks(
    video_path,
    output_dir="./chunks",
    chunk_size=100,
    debug=True
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if debug:
        print(f"FPS: {fps}")
        print(f"Resolution: {width}x{height}")
        print(f"Total frames: {total_frames}")
        print(f"Chunk size: {chunk_size}\n")

    # ✅ ONE progress bar, frames only
    pbar = tqdm(
        total=total_frames,
        desc="Processing frames",
        unit="frame"
    )

    chunk_index = 1
    frame_in_chunk = 0
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Open new chunk silently
        if writer is None:
            chunk_path = os.path.join(output_dir, f"{chunk_index:04d}.mp4")
            writer = cv2.VideoWriter(
                chunk_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        # ---- YOUR HEAVY PROCESSING ----
        try:
            image, result, result_path = local_ocr(
                frame,
                bounding_box=False,
                hide_method=1
            )
        except Exception:
            result = frame

        writer.write(result)
        frame_in_chunk += 1
        pbar.update(1)

        # Close chunk silently
        if frame_in_chunk >= chunk_size:
            writer.release()
            writer = None
            frame_in_chunk = 0
            chunk_index += 1

    if writer is not None:
        writer.release()

    cap.release()
    pbar.close()

    if debug:
        print(f"\n✔ Done. Video written in chunks at: {output_dir}")



import os
import subprocess


def is_valid_video(video_path):
    """
    Check if FFmpeg can read the video file.
    Returns True if valid, False if corrupted.
    """
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path, "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0


def concat_videos_with_original_audio(
    original_video,
    chunks_dir="./chunks",
):
    # -----------------------------
    # Normalize paths
    # -----------------------------
    original_video = os.path.abspath(original_video)
    chunks_dir = os.path.abspath(chunks_dir)

    os.makedirs("./save_videos", exist_ok=True)

    # -----------------------------
    # Output path
    # ./save_videos/{name}_no_text.{ext}
    # -----------------------------
    base = os.path.basename(original_video)
    name, ext = os.path.splitext(base)
    output_path = os.path.abspath(f"./save_videos/{name}_no_text{ext}")

    # -----------------------------
    # Collect & sort chunks
    # -----------------------------
    all_chunks = sorted(
        f for f in os.listdir(chunks_dir)
        if f.lower().endswith(".mp4")
    )

    if not all_chunks:
        print("❌ No video chunks found")
        return

    # -----------------------------
    # Validate chunks
    # -----------------------------
    valid_chunks = []
    for chunk in all_chunks:
        path = os.path.join(chunks_dir, chunk)
        if is_valid_video(path):
            valid_chunks.append(chunk)
        else:
            print(f"⚠ Skipping corrupted chunk: {chunk}")

    if not valid_chunks:
        print("❌ All chunks are corrupted")
        return

    # -----------------------------
    # Create concat list
    # -----------------------------
    concat_list = os.path.join(chunks_dir, "concat_list.txt")

    with open(concat_list, "w", encoding="utf-8") as f:
        for chunk in valid_chunks:
            p = os.path.join(chunks_dir, chunk).replace("\\", "/")
            f.write(f"file '{p}'\n")

    # -----------------------------
    # Concatenate video (no audio)
    # -----------------------------
    temp_video = os.path.join(chunks_dir, "video_no_audio.mp4")

    concat_proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            temp_video
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if concat_proc.returncode != 0:
        print("❌ FFmpeg concat failed:")
        print(concat_proc.stderr)
        return

    # -----------------------------
    # Extract audio from original (WAV)
    # -----------------------------
    temp_audio = os.path.join(chunks_dir, "original_audio.wav")

    audio_proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", original_video,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            temp_audio
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if audio_proc.returncode != 0:
        print("❌ FFmpeg audio extraction failed:")
        print(audio_proc.stderr)
        return

    # -----------------------------
    # Mux video + audio (SILENT)
    # -----------------------------
    mux_proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", temp_video,
            "-i", temp_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if mux_proc.returncode != 0:
        print("❌ FFmpeg muxing failed:")
        print(mux_proc.stderr)
        return

    # -----------------------------
    # Cleanup
    # -----------------------------
    for f in [concat_list, temp_video, temp_audio]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\n✔ FINAL VIDEO READY: {output_path}")
    print(f"✔ Used {len(valid_chunks)} valid chunks")


# -----------------------------
# RUN
# -----------------------------

video_path="./test.mp4"
extract_and_write_chunks(
    video_path,
    output_dir="./chunks",
    chunk_size=100,
    debug=True
)

concat_videos_with_original_audio(
    original_video=video_path,
    chunks_dir="./chunks"
)
