import os
import sys
import shutil
import subprocess
import cv2
import torch
import numpy as np
import gradio as gr
from tqdm.auto import tqdm

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
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
        
    ocr_items = run_ocr(img)
    
    # Draw boxes on raw image copy for visualization
    raw_img_vis = img.copy()
    if bounding_box:
        raw_img_vis = draw_polygons(raw_img_vis, ocr_items, display_text=False, expand_scale=expand)

    # Generate Result
    if hide_method == 1: # Inpaint
        mask = generate_text_mask(img, ocr_items, invert=False, min_conf=0.0, expand_scale=expand)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # MIGAN expects RGB, returns RGB (usually), ensure we handle formats correctly
        output_inpaint = model(img_rgb, mask, config)
        # Convert back to BGR for OpenCV consistency if model returns RGB
        if output_inpaint.shape[-1] == 3:
             result = cv2.cvtColor(output_inpaint, cv2.COLOR_RGB2BGR)
        else:
             result = output_inpaint
        temp_mask = mask
    else: # Blur
        result = blur_text_regions(img, ocr_items, blur_level=3, blur_sigma=0, min_conf=0.0, expand_scale=expand)
        temp_mask = np.zeros_like(img[:,:,0]) # Placeholder mask

    return raw_img_vis, result, temp_mask

def display_result(image, mask, result, scale=0.4, out_size=(1280, 720)):
    """
    Creates the 2x2 grid visualization.
    """
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    pw = int(w * scale)
    ph = int(h * scale)

    ocr_r    = cv2.resize(image,  (pw, ph))
    mask_r   = cv2.resize(mask,   (pw, ph))
    result_r = cv2.resize(result, (pw, ph))
    # result_r=rgb to bgr convert 
    result_r = cv2.cvtColor(result_r, cv2.COLOR_RGB2BGR)
    canvas_w = pw * 2
    canvas_h = ph * 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Layout
    canvas[0:ph, 0:pw] = ocr_r
    canvas[0:ph, pw:canvas_w] = mask_r
    
    # Bottom centered
    x_start = (canvas_w - pw) // 2
    canvas[ph:ph + ph, x_start:x_start + pw] = result_r

    # Labels
    cv2.putText(canvas, "OCR DETECT", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canvas, "GENERATED MASK", (pw + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(canvas, "INPAINTED RESULT", (x_start + 15, ph + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Resize to specific output size for UI consistency
    canvas_final = cv2.resize(canvas, out_size)
    
    # Gradio expects RGB images
    return cv2.cvtColor(canvas_final, cv2.COLOR_BGR2RGB)

# --- 4. Video Processing Logic ---

def cleanup_dirs(chunks_dir):
    if os.path.exists(chunks_dir):
        shutil.rmtree(chunks_dir)
    os.makedirs(chunks_dir, exist_ok=True)

def concat_videos_with_original_audio(original_video, chunks_dir):
    """
    Concatenates chunks and merges with original audio using FFmpeg.
    """
    original_video = os.path.abspath(original_video)
    chunks_dir = os.path.abspath(chunks_dir)
    save_dir = os.path.abspath("./save_videos")
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.basename(original_video)
    name, ext = os.path.splitext(base)
    output_path = os.path.join(save_dir, f"{name}_cleaned{ext}")

    # Collect chunks
    all_chunks = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".mp4")])
    if not all_chunks:
        return None

    # Create list file
    list_path = os.path.join(chunks_dir, "mylist.txt")
    with open(list_path, "w") as f:
        for chunk in all_chunks:
            p = os.path.join(chunks_dir, chunk).replace("\\", "/")
            f.write(f"file '{p}'\n")

    # 1. Concat video
    temp_video = os.path.join(chunks_dir, "temp_no_audio.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, 
        "-c", "copy", temp_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Extract Audio
    temp_audio = os.path.join(chunks_dir, "audio.aac")
    subprocess.run([
        "ffmpeg", "-y", "-i", original_video, "-vn", "-acodec", "copy", temp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Merge (if audio exists, otherwise just copy video)
    if os.path.exists(temp_audio):
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_video, "-i", temp_audio, 
            "-c:v", "copy", "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0", 
            "-shortest", output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copy(temp_video, output_path)

    return output_path

def process_video_pipeline(video_path,hide_method="Inpaint",expand=1.1):
    """
    Main Generator Function for Gradio.
    """
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

    chunk_size = 60 # Write to disk every 60 frames
    chunk_index = 0
    frames_buffer = []
    
    pbar = tqdm(total=total_frames, desc="Processing")

    chunk_writer = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- Core Processing ---
        if hide_method == "Inpaint":
            hide_method_num = 1
        else:
            hide_method_num = 2
        try:
            # Expand=1.1 gives slightly better coverage for text removal
            vis_img, clean_img, mask = local_ocr(frame, bounding_box=True, hide_method=hide_method_num, expand=expand)
        except Exception as e:
            print(f"Frame error: {e}")
            vis_img, clean_img, mask = frame, frame, np.zeros_like(frame[:,:,0])

        # --- Create Visualization for Gradio Stream ---
        # Note: We return RGB for Gradio display
        preview_frame = display_result(vis_img, mask, clean_img,scale=0.4)#, out_size=(640, 360))

        # --- Write Result to Chunk ---
        if chunk_writer is None:
            chunk_name = os.path.join(chunks_dir, f"{chunk_index:05d}.mp4")
            chunk_writer = cv2.VideoWriter(
                chunk_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        
        chunk_writer.write(clean_img)
        
        # Handle chunk rotation
        if len(frames_buffer) >= chunk_size:
            chunk_writer.release()
            chunk_writer = None
            chunk_index += 1
            frames_buffer = [] # Reset buffer counter logic if needed, currently using write direct
        
        pbar.update(1)
        
        # YIELD to Gradio: (Image Update, None for final video)
        yield preview_frame, None

    # Cleanup last chunk
    if chunk_writer is not None:
        chunk_writer.release()
    
    cap.release()
    pbar.close()

    # --- Final Concatenation ---
    print("Concatenating video and merging audio...")
    final_output_path = concat_videos_with_original_audio(video_path, chunks_dir)
    
    # Return last preview frame and the path to the final video
    yield preview_frame, final_output_path

def subtitle_remove_ui():
    custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""
    with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
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
                # We use Image for the live stream because it's smoother for frame-by-frame updates
                final_video_output = gr.Video(label="Final Output Video (Downloadable)")
                with gr.Accordion("Live Processing View", open=False):
                    live_display = gr.Image(label="Live Processing View (OCR | Mask | Result)", interactive=False)
                
                
        
            
            

        # Connect the generator
        btn_run.click(
            fn=process_video_pipeline,
            inputs=[input_video, hide_method, bbox_expand],
            outputs=[live_display, final_video_output]
        )
        return demo

# demo = subtitle_remove_ui()
# demo.launch()
    

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo=subtitle_remove_ui()
    demo.queue().launch(debug=debug, share=share)
if __name__ == "__main__":
    main()        
