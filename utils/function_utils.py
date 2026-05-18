import cv2
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.colors  import ListedColormap
from PIL import Image

def heatmap_overlay(img, attn, colormap=cv2.COLORMAP_JET, intensity=0.85, gamma=0.4, gamma_threshold=0.7):

    attn_gray = attn.mean(axis=2) if attn.ndim == 3 else attn
    attn_normalized = (attn_gray - attn_gray.min()) / (attn_gray.max() - attn_gray.min() + 1e-8)

    attn_gamma_global = np.power(attn_normalized, gamma)
    attn_final_for_heatmap = np.copy(attn_normalized)

    mask = attn_normalized > gamma_threshold
    attn_final_for_heatmap[mask] = attn_gamma_global[mask]

    heatmap_ = cv2.applyColorMap((attn_final_for_heatmap * 255).astype(np.uint8), colormap)
    heatmap_ = cv2.cvtColor(heatmap_, cv2.COLOR_BGR2RGB)
    heatmap = heatmap_ / 255.0

    if img.max() > 1: img = img.astype(float) / 255.0

    alpha = attn_normalized[..., np.newaxis] * intensity
    blended = img * (1 - alpha) + heatmap * alpha

    return np.clip(blended, 0, 1), heatmap_

def classification_overlay(arr:np.ndarray, save_pth:str):

    cmap_colors = np.array([
        [89.0/255.0,     24.0/255.0,     126.0/255.0],
        [247.0 / 255.0,   221.0/255.0,   47.0/255.0],
        [103.0/255.0,   169.0/255.0,   204.0/255.0],
        [1.0, 1.0, 1.0]
    ])

    cmap = ListedColormap(cmap_colors)

    rgb_array = (cmap(arr)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb_array).save(save_pth)

    return cmap(arr)[...,:3]

def intensity_mixture(raw_wsi, intensity_map, cls_map,save_pth, degree=0.4):

    alpha = intensity_map[..., np.newaxis] * degree
    blended = raw_wsi * (1 - alpha) + (cls_map * 255) * alpha #0~255

    Image.fromarray(blended.astype(np.uint8)).save(save_pth)

