import os
import sys
import shutil
import subprocess
import cv2
import torch
import numpy as np
import gradio as gr
from tqdm.auto import tqdm
import time
import click

# --- 1. Setup Paths & Imports ---
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

# Try imports to ensure environment is correct
try:
    from mi_gan.image_inpainting import MIGAN
    from mi_gan.schema import InpaintRequest
    from ocr.utils import run_ocr, draw_polygons, generate_text_mask, blur_text_regions
except ImportError as e:
    print("Error importing modules. Make sure 'mi_gan' and 'ocr' folders are in the current directory.")
    raise e

# --- 2. Initialize Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading MIGAN model on {device}...")
model = MIGAN(device=device)
config = InpaintRequest(
    hd_strategy="crop",
    hd_strategy_crop_margin=128,
)
print("Model loaded.")


# --- 3. Helper Functions (OCR & Display) ---

def local_ocr(image, bounding_box=True, hide_method=1, expand=1.0):
    """
    Processes a single frame: Detects text, creates mask, inpaints/blurs.
    Ensures output is ALWAYS strictly in BGR format for OpenCV to save correctly.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
        
    ocr_items = run_ocr(img)
    
    # Draw boxes on raw image copy for visualization (Current format: BGR)
    raw_img_vis = img.copy()
    if bounding_box:
        raw_img_vis = draw_polygons(raw_img_vis, ocr_items, display_text=False, expand_scale=expand)

    # Generate Result
    if hide_method == 1: # Inpaint
        mask = generate_text_mask(img, ocr_items, invert=False, min_conf=0.0, expand_scale=expand)
        
        # MIGAN usually expects RGB input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_inpaint = model(img_rgb, mask, config)
        
        # --- FOOLPROOF COLOR SPACE AUTO-DETECTOR ---
        # Some models return RGB, some return BGR internally.
        # We mathematically check which one it returned so we never save a blue video!
        out_float = output_inpaint.astype(np.float32)
        
        if out_float.shape == img.shape:
            mse_bgr = np.mean((out_float - img.astype(np.float32)) ** 2)
            mse_rgb = np.mean((out_float - img_rgb.astype(np.float32)) ** 2)
            
            if mse_rgb < mse_bgr:
                # The model returned RGB! We convert to BGR.
                result = cv2.cvtColor(output_inpaint, cv2.COLOR_RGB2BGR)
            else:
                # The model returned BGR! Do NOT convert it.
                result = output_inpaint
        else:
            # Fallback if shapes differ
            result = cv2.cvtColor(output_inpaint, cv2.COLOR_RGB2BGR) if output_inpaint.shape[-1] == 3 else output_inpaint
             
        temp_mask = mask
    else: # Blur (OpenCV handles natively in BGR)
        result = blur_text_regions(img, ocr_items, blur_level=3, blur_sigma=0, min_conf=0.0, expand_scale=expand)
        temp_mask = np.zeros_like(img[:,:,0]) # Placeholder mask

    # Return raw_img_vis (BGR), result (BGR), and mask (Grayscale)
    return raw_img_vis, result, temp_mask


def display_result(image, mask, result, scale=0.4, out_size=(1280, 720)):
    """
    Creates the 2x2 grid visualization fitting into a fixed aspect ratio canvas.
    Images will be letterboxed/pillarboxed to prevent stretching.
    """
    # Ensure mask is 3-channel (BGR)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Create the fixed final blank canvas
    out_w, out_h = out_size
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    box_w = out_w // 2
    box_h = out_h // 2

    def fit_image_to_box(img, target_w, target_h):
        h, w = img.shape[:2]
        fit_scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * fit_scale), int(h * fit_scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        box_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        box_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return box_canvas

    # ALL inputs here are safely guaranteed to be BGR now. No need for color swapping!
    ocr_fit    = fit_image_to_box(image, box_w, box_h)
    mask_fit   = fit_image_to_box(mask, box_w, box_h)
    result_fit = fit_image_to_box(result, box_w, box_h)

    # Layout Placement
    canvas[0:box_h, 0:box_w] = ocr_fit                 # Top-Left: OCR
    canvas[0:box_h, box_w:out_w] = mask_fit            # Top-Right: Mask
    
    x_start = (out_w - box_w) // 2
    canvas[box_h:out_h, x_start:x_start+box_w] = result_fit # Bottom-Center: Result

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Text Shadows (Black)
    cv2.putText(canvas, "OCR DETECT", (15, 35), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, "GENERATED MASK", (box_w + 15, 35), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, "INPAINTED RESULT", (x_start + 15, box_h + 35), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    
    # Actual Text Color
    cv2.putText(canvas, "OCR DETECT", (15, 35), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(canvas, "GENERATED MASK", (box_w + 15, 35), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(canvas, "INPAINTED RESULT", (x_start + 15, box_h + 35), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    # Convert the completely built BGR canvas to RGB solely for Gradio's web display
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def put_text_bottom_right(frame, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(255, 255, 255), thickness=2, margin=10, shadow=True):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = frame.shape[1] - text_w - margin
    y = frame.shape[0] - margin

    if shadow:
        cv2.putText(frame, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


# --- 4. Video Processing Logic ---

def cleanup_dirs(chunks_dir):
    if os.path.exists(chunks_dir):
        shutil.rmtree(chunks_dir)
    os.makedirs(chunks_dir, exist_ok=True)


def concat_videos_with_original_audio(original_video, chunks_dir):
    original_video = os.path.abspath(original_video)
    chunks_dir = os.path.abspath(chunks_dir)
    save_dir = os.path.abspath("./save_videos")
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.basename(original_video)
    name, ext = os.path.splitext(base)
    output_path = os.path.join(save_dir, f"{name}_cleaned{ext}")

    all_chunks = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".mp4")])
    if not all_chunks:
        return None

    list_path = os.path.join(chunks_dir, "mylist.txt")
    with open(list_path, "w") as f:
        for chunk in all_chunks:
            p = os.path.join(chunks_dir, chunk).replace("\\", "/")
            f.write(f"file '{p}'\n")

    temp_video = os.path.join(chunks_dir, "temp_no_audio.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, 
        "-c", "copy", temp_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    temp_audio = os.path.join(chunks_dir, "audio.aac")
    subprocess.run([
        "ffmpeg", "-y", "-i", original_video, "-vn", "-acodec", "copy", temp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp_audio):
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_video, "-i", temp_audio, 
            "-c:v", "copy", "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0", 
            "-shortest", output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copy(temp_video, output_path)

    # Google Drive Integration
    if os.path.exists("/content/gdrive/MyDrive"):
        drive_folder="/content/gdrive/MyDrive/subtitle_inpaint"
        try:
            os.makedirs(drive_folder, exist_ok=True)
            time.sleep(1)  
            shutil.copy(output_path, drive_folder)
            print(f"Copied output video to Google Drive: {drive_folder}")
            gr.Warning(f"Copied output video to Google Drive: {drive_folder}", duration=5)
        except Exception as e:
            print(f"Could not copy to Google Drive: {e}")
        
    return output_path


def process_video_pipeline(video_path, hide_method="Inpaint", expand=1.1):
    if not video_path:
        return None, None

    chunks_dir = "./chunks"
    cleanup_dirs(chunks_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    chunk_size = 60 
    chunk_index = 0
    frames_buffer = []
    
    pbar = tqdm(total=total_frames, desc="Processing")

    chunk_writer = None
    frame_id = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hide_method_num = 1 if hide_method == "Inpaint" else 2
        
        try:
            # Outputs returned from here are guaranteed 100% standard BGR format
            vis_img, clean_img, mask = local_ocr(frame, bounding_box=True, hide_method=hide_method_num, expand=expand)
        except Exception as e:
            print(f"Frame error: {e}")
            vis_img, clean_img, mask = frame, frame, np.zeros_like(frame[:,:,0])

        # Create Gradio Visuals
        preview_frame = display_result(vis_img, mask, clean_img, scale=0.4)
        display_text = f"Frame {frame_id}/{total_frames}"
        frame_id += 1
        preview_frame = put_text_bottom_right(preview_frame, display_text)

        # Write clean_img (BGR) to OpenCV chunk writer
        if chunk_writer is None:
            chunk_name = os.path.join(chunks_dir, f"{chunk_index:05d}.mp4")
            chunk_writer = cv2.VideoWriter(
                chunk_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        
        chunk_writer.write(clean_img)
        frames_buffer.append(True)
        
        if len(frames_buffer) >= chunk_size:
            chunk_writer.release()
            chunk_writer = None
            chunk_index += 1
            frames_buffer = []
        
        pbar.update(1)
        yield preview_frame, None

    # Cleanup
    if chunk_writer is not None:
        chunk_writer.release()
    
    cap.release()
    pbar.close()

    print("Concatenating video and merging audio...")
    final_output_path = concat_videos_with_original_audio(video_path, chunks_dir)
    
    yield preview_frame, final_output_path


# --- 5. Gradio UI ---

def subtitle_remove_ui():
    custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML("""
            <div style="text-align: center; margin: 20px auto; max-width: 800px;">
                <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎥 AI Video Text Remover (Image Inpainting)</h1>
            </div>""")        
        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.File(label="Upload Video")
                btn_run = gr.Button("🚀 Start Processing", variant="primary")
                with gr.Accordion("Settings", open=False):
                    hide_method = gr.Radio(
                        choices=["Inpaint", "Blur"],
                        value="Inpaint",
                        label="Text Removal Method",
                        info="Choose 'Inpaint' to remove text using AI inpainting or 'Blur' to blur text regions."
                    )
                    bbox_expand = gr.Slider(
                        minimum=1.0,
                        maximum=4.0,
                        value=2.0,
                        step=0.1,   
                        label="Bounding Box Expand Scale",
                        info="Scale to expand the detected text bounding boxes for better coverage."
                    )
            with gr.Column(scale=2):
                final_video_output = gr.File(label="Download Processed Video")
                with gr.Accordion("Live Processing View", open=True):
                    live_display = gr.Image(label="Live Processing View (OCR | Mask | Result)", interactive=False)
                
        # Connect generator
        btn_run.click(
            fn=process_video_pipeline,
            inputs=[input_video, hide_method, bbox_expand],
            outputs=[live_display, final_video_output]
        )
        return demo


@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo = subtitle_remove_ui()
    demo.queue().launch(debug=debug, share=share)

if __name__ == "__main__":
    main()
