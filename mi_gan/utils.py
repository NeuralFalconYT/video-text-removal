import os
import sys
import cv2
import torch
import hashlib
from urllib.parse import urlparse
from typing import List
from loguru import logger
from typing import Optional, Literal, List
import numpy as np
from torch.hub import download_url_to_file, get_dir

def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def switch_mps_device(model_name, device):
    MPS_UNSUPPORT_MODELS = [
    "lama",
    "ldm",
    "zits",
    "mat",
    "fcf",
    "cv2",
    "manga",
    ]
    if model_name in MPS_UNSUPPORT_MODELS and str(device) == "mps":
        logger.info(f"{model_name} not support mps, switch to cpu")
        return torch.device("cpu")
    return device
# -------------------------------------------------
# Model download & cache
# -------------------------------------------------

def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    return os.path.join(model_dir, filename)


def download_model(url, model_md5: str = None):
    if os.path.exists(url):
        cached_file = url
    else:
        cached_file = get_cache_path_by_url(url)

    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, None, progress=True)

        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 != _md5:
                try:
                    os.remove(cached_file)
                except Exception:
                    pass
                logger.error(
                    f"Model md5: {_md5}, expected md5: {model_md5}. "
                    f"Please delete {cached_file} and retry."
                )
                exit(-1)
            logger.info(f"Download model success, md5: {_md5}")

    return cached_file


def load_jit_model(url_or_path, device, model_md5: str):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    logger.info(f"Loading model from: {model_path}")
    model = torch.jit.load(model_path, map_location="cpu").to(device)
    model.eval()
    return model


# -------------------------------------------------
# Image helpers
# -------------------------------------------------

def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    return np_img.astype("float32") / 255.0


def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, (new_w, new_h), interpolation=interpolation)
    return np_img


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (H, W) or (H, W, 1), 0~255

    Returns:
        list of [x1, y1, x2, y2]
    """
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h], dtype=int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )
