# %%writefile /content/video-text-removal/ocr/utils.py
from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np

DET = "./ocr/models/ch_PP-OCRv5_mobile_det.onnx"
CLS = "./ocr/models/ch_ppocr_mobile_v2.0_cls_infer.onnx"
REC_CN = "./ocr/models/ch_PP-OCRv5_rec_mobile_infer.onnx"

ocr = RapidOCR(
    use_cuda=False,
    det_model_path=DET,
    rec_model_path=REC_CN,   # Chinese model handles English too
    cls_model_path=CLS,
)



from PIL import Image, ImageDraw, ImageFont

font_path = "./ocr/fonts/simfang.ttf"
font = ImageFont.truetype(font_path, 28)
def draw_polygons(image, ocr_items, display_text=False):
    # OpenCV → PIL
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for item in ocr_items:
        poly = np.array(item["polygon"], dtype=np.int32)

        # draw polygon (outline)
        draw.line(
            [tuple(p) for p in poly] + [tuple(poly[0])],
            width=2,
            fill=(0, 255, 0)
        )

        if display_text:
            x, y = poly[0]
            draw.text(
                (x, max(y - 30, 0)),
                item["text"],
                font=font,
                fill=(255, 0, 0)
            )

    # PIL → OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
# import cv2
# import numpy as np

# def run_ocr(image):
#   results, elapse = ocr(image)
#   ocr_items = []
#   def is_chinese(text):
#       return any('\u4e00' <= c <= '\u9fff' for c in text)
#   for poly, text, score in results:
#       ocr_items.append({
#           "polygon": poly,
#           "text": text,
#           "confidence": float(score)
#       })
#   for item in ocr_items:
#       item["lang"] = "ch" if is_chinese(item["text"]) else "en"
#   return ocr_items

import cv2
import numpy as np

def run_ocr(image):
    ocr_items = []

    try:
        results, elapse = ocr(image)
        if results is None:
            return []

        def is_chinese(text):
            return any('\u4e00' <= c <= '\u9fff' for c in text)

        for item in results:
            # extra safety
            if len(item) != 3:
                continue

            poly, text, score = item
            ocr_items.append({
                "polygon": poly,
                "text": text,
                "confidence": float(score),
                "lang": "ch" if is_chinese(text) else "en"
            })

    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return []

    return ocr_items



def generate_text_mask(
    image,
    ocr_items,
    invert=False,
    min_conf=0.0,
    expand_scale=1.0
):
    """
    image        : OpenCV image (H, W, C)
    ocr_items    : list of dicts with 'polygon'
    invert       : False -> text=white, bg=black
                   True  -> text=black, bg=white
    min_conf     : confidence threshold
    expand_scale : expand polygon size (1.0 = no expand)
    """

    h, w = image.shape[:2]

    # background value
    bg_val = 255 if invert else 0
    fg_val = 0 if invert else 255

    mask = np.full((h, w), bg_val, dtype=np.uint8)

    for item in ocr_items:
        if item.get("confidence", 1.0) < min_conf:
            continue

        poly = np.array(item["polygon"], dtype=np.float32)

        # optional polygon expansion
        if expand_scale != 1.0:
            center = poly.mean(axis=0)
            poly = (poly - center) * expand_scale + center

        poly = poly.astype(np.int32)

        cv2.fillPoly(mask, [poly], fg_val)

    return mask



def blur_text_regions(
    image,
    ocr_items,
    blur_level=3,
    blur_sigma=0,
    min_conf=0.0,
    expand_scale=1.0
):
    """
    image        : OpenCV image (H, W, C)
    ocr_items    : OCR items with 'polygon'
    blur_level   : 1 (light) → 4 (very strong)
    blur_sigma   : Gaussian sigma (0 = auto)
    min_conf     : confidence threshold
    expand_scale : expand polygon (1.0 = no expand)
    """

    # Clamp blur level safely
    blur_level = int(np.clip(blur_level, 1, 4))

    kernel_map = {
        1: (31, 31),    # light
        2: (81, 81),    # strong
        3: (99, 99),    # very strong
        4: (151, 151),  # extreme
    }

    blur_ksize = kernel_map[blur_level]

    h, w = image.shape[:2]
    output = image.copy()

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)

    for item in ocr_items:
        if item.get("confidence", 1.0) < min_conf:
            continue

        poly = np.array(item["polygon"], dtype=np.float32)

        # Optional expansion
        if expand_scale != 1.0:
            center = poly.mean(axis=0)
            poly = (poly - center) * expand_scale + center

        poly = poly.astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)

    # Blur full image ONCE (important for speed)
    blurred_full = cv2.GaussianBlur(image, blur_ksize, blur_sigma)

    # Apply blur only where text exists
    output[mask == 255] = blurred_full[mask == 255]

    return output
