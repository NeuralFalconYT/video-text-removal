# import sys
# path = "/content/video-text-removal"

# if path not in sys.path:
#     sys.path.insert(0, path)

from mi_gan.image_inpainting import MIGAN
from mi_gan.schema import InpaintRequest
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MIGAN(device=device)
config = InpaintRequest(
    hd_strategy="crop",
    hd_strategy_crop_margin=128,
)

import cv2 # Moved import cv2 here to be consistent with usage
img_bgr = cv2.imread("/content/frame.png")
mask = cv2.imread("/content/mask.png", cv2.IMREAD_GRAYSCALE)

# Convert image to RGB (VERY IMPORTANT)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
output_bgr = model(img_rgb, mask, config)
cv2.imwrite("/content/output.png", output_bgr)
# from google.colab.patches import cv2_imshow
# cv2_imshow(output_bgr)
