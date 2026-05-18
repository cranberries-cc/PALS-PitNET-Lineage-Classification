import matplotlib.pyplot  as plt
import networkx as nx
import numpy as np
import os
from torch_geometric.utils  import to_networkx
from .file_utils import find_target_wsi_img
from .data_utils import visualize_heatmap_new
from .function_utils import heatmap_overlay
from .function_utils import classification_overlay
from .function_utils import intensity_mixture
import cv2
from PIL import Image
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import torch

def visualize_attention_v2(single_input, model, args, slide_id,results_fold_pth,pos_array,cluster_indices, gt, pre_label, pre_confi):

    save_pth = os.path.join(results_fold_pth,'interpre_vis')
    os.makedirs(save_pth,exist_ok=True)

    node_atten = model.node_importance_scores

    major_id, sub_id = int(slide_id[0][0]), int(slide_id[0][1])
    target_wsi = find_target_wsi_img(major_id, sub_id,args.wsi_pth)

    level0_w, level0_h = target_wsi.level_dimensions[0]
    level3_w, level3_h = target_wsi.level_dimensions[3]

    region = target_wsi.read_region(location=(0, 0), level=3, size=(level3_w, level3_h))
    region_rgb = region.convert("RGB")
    raw_wsi = np.array(region_rgb)

    draw_node_attention(raw_wsi, level0_w, level0_h, level3_w, level3_h,save_pth, node_atten, pos_array,
                        f'{major_id}_{sub_id}_attention_map_GT{str(gt)}_PRE{str(pre_label)}_CONFI_{str(pre_confi)}', args.mag,cluster_indices,major_id, sub_id,args,_is_save_raw=False)


def draw_node_attention(raw_wsi, w0, h0, w3, h3,save_pth,node_atten,all_patchs_pos,atten_name,mag,cluster_indices,major_id, sub_id,args,_is_save_raw=False):

    pre_hot_map_full_res = np.zeros((h0, w0), dtype=np.float32)
    patch_size_level0 = int(512 * (40 / mag))

    for node_att, node_id in zip(node_atten, range(len(node_atten))):
        node_patch_indices = cluster_indices[0][node_id]
        node_patch_pos = all_patchs_pos[0][node_patch_indices]
        for pos in node_patch_pos:
            x_level0_col, y_level0_row = pos[0], pos[1]
            row_start = y_level0_row
            row_end = min(y_level0_row + patch_size_level0, h0)
            col_start = x_level0_col
            col_end = min(x_level0_col + patch_size_level0, w0)
            pre_hot_map_full_res[int(row_start):int(row_end), int(col_start):int(col_end)] = node_att.cpu().numpy()

    sel_high_low_attn_patches = False
    if sel_high_low_attn_patches:
        high_attn_patch_dir = os.path.join(save_pth,f"{major_id}_{sub_id}_sel_high_low_attn_patches","highattn")
        low_attn_patch_dir = os.path.join(save_pth,f"{major_id}_{sub_id}_sel_high_low_attn_patches","lowattn")
        os.makedirs(high_attn_patch_dir,exist_ok=True)
        os.makedirs(low_attn_patch_dir,exist_ok=True)
        ori_wsi = find_target_wsi_img(major_id, sub_id,args.wsi_pth)
        top5_max_vals = torch.topk(node_atten, k=10, dim=0, largest=True, sorted=True)
        top5_min_vals = torch.topk(node_atten, k=10, dim=0, largest=False, sorted=True)

        for node_att, node_id in zip(node_atten, range(len(node_atten))):
            node_patch_indices = cluster_indices[0][node_id]
            node_patch_pos = all_patchs_pos[0][node_patch_indices]
            if node_att >= top5_max_vals.values.min():
               cur_save_pth =  high_attn_patch_dir
            elif node_att <= top5_min_vals.values.max():
                cur_save_pth = low_attn_patch_dir
            else:
                continue
            for idx, pos in enumerate(node_patch_pos):
                x_level0_col, y_level0_row = pos[0], pos[1] #
                lel = 0
                region = ori_wsi.read_region(location=(x_level0_col, y_level0_row), level=lel, size=(512, 512)).convert("RGB")
                region.save(os.path.join(cur_save_pth,f"40×_cluster_{node_id}_patch{idx}.png"))

    scale_factor = w0 / w3
    pre_hot_map_resized = cv2.resize(pre_hot_map_full_res, (w3, h3), interpolation=cv2.INTER_AREA)
    background_mask_resized = (pre_hot_map_resized == 0).astype(np.uint8)

    pre_hot_map_resized = gaussian_filter(pre_hot_map_resized, sigma=87/scale_factor)
    pre_hot_map_resized[background_mask_resized.astype(bool)] = 0.0

    attn_normalized = (pre_hot_map_resized - pre_hot_map_resized.min()) / (pre_hot_map_resized.max() - pre_hot_map_resized.min() + 1e-8)
    heat_map = cv2.applyColorMap((attn_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat_map).save(os.path.join(save_pth, f"{atten_name}_heatmap.png"))

def visualize_multiple_synchronization(args, slide_id,results_fold_pth,pos_array,cluster_indices,linear_preds_full, linear_probs_full ,linear_preds_partial, selected_indices, tresh=0.5):

    major_id, sub_id = int(slide_id[0]), int(slide_id[1])
    target_wsi = find_target_wsi_img(major_id, sub_id, args.wsi_pth)
    if target_wsi is None:
        return False

    level0_w, level0_h = target_wsi.level_dimensions[0]
    level3_w, level3_h = target_wsi.level_dimensions[3]

    region = target_wsi.read_region(location=(0, 0), level=3, size=(level3_w, level3_h))
    region_rgb = region.convert("RGB")
    raw_wsi = np.array(region_rgb)

    patch_size_level0 = 512
    classification_map_full = np.full((level0_h * 2,  level0_w * 2), fill_value=3, dtype=np.uint8)
    intensity_map_full = np.full((level0_h * 2, level0_w * 2), fill_value=0, dtype=np.uint8)
    for node_cls, node_id in zip(linear_preds_full,range(len(linear_preds_full))):
        node_patch_indices = cluster_indices[node_id]
        node_patch_pos = pos_array[node_patch_indices]
        for pos in node_patch_pos:
            x_level0_col, y_level0_row = pos[0], pos[1]
            row_start = y_level0_row
            row_end = y_level0_row + patch_size_level0
            col_start = x_level0_col
            col_end = x_level0_col + patch_size_level0
            classification_map_full[int(row_start):int(row_end), int(col_start):int(col_end)] = node_cls.cpu().numpy().astype(np.uint8)
            intensity_map_full[int(row_start):int(row_end), int(col_start):int(col_end)] = 1

    classification_map_full_resized = cv2.resize(classification_map_full, (level3_w, level3_h), interpolation=cv2.INTER_AREA)
    intensity_map_full_resized = cv2.resize(intensity_map_full, (level3_w, level3_h), interpolation=cv2.INTER_AREA)
    map_full = classification_overlay(classification_map_full_resized,os.path.join(results_fold_pth, f"{slide_id[0]}_{slide_id[1]}_multisynchron_map_full_notreshold.png"))

    classification_map_full_treshed = np.full((level0_h * 2, level0_w * 2), fill_value=3, dtype=np.uint8)
    intensity_map_full_treshed = np.full((level0_h * 2, level0_w * 2), fill_value=0, dtype=np.uint8)
    for node_cls, node_id, node_prob in zip(linear_preds_full, range(len(linear_preds_full)),linear_probs_full):
        if node_prob.max() < tresh:
            continue
        node_patch_indices = cluster_indices[node_id]
        node_patch_pos = pos_array[node_patch_indices]
        for pos in node_patch_pos:
            x_level0_col, y_level0_row = pos[0], pos[1]
            row_start = y_level0_row

            row_end = y_level0_row + patch_size_level0
            col_start = x_level0_col

            col_end = x_level0_col + patch_size_level0
            classification_map_full_treshed[int(row_start):int(row_end),int(col_start):int(col_end)] = node_cls.cpu().numpy().astype(np.uint8)
            intensity_map_full_treshed[int(row_start):int(row_end),int(col_start):int(col_end)] = 1

    classification_map_full_treshed_resized = cv2.resize(classification_map_full_treshed, (level3_w, level3_h),
                                                 interpolation=cv2.INTER_AREA)
    intensity_map_full_treshed_resized = cv2.resize(intensity_map_full_treshed, (level3_w, level3_h),
                                                 interpolation=cv2.INTER_AREA)
    map_full_treshed = classification_overlay(classification_map_full_treshed_resized, os.path.join(results_fold_pth,
                                                                         f"{slide_id[0]}_{slide_id[1]}_multisynchron_map_full_tresh{tresh}.png"))

    return True


def visualize_Multi_hormone(args, slide_id,results_fold_pth,pos_array,cluster_indices,linear_preds_full, linear_probs_full ,linear_preds_partial, selected_indices, tresh=0.5):

    major_id, sub_id = int(slide_id[0]), int(slide_id[1])
    target_wsi = find_target_wsi_img(major_id, sub_id, args.wsi_pth)
    if target_wsi is None:
        return False
    level0_w, level0_h = target_wsi.level_dimensions[0]
    level3_w, level3_h = target_wsi.level_dimensions[3]

    sf1_prob_map = np.zeros((level0_h, level0_w), dtype=np.float32)
    pit1_prob_map = np.zeros((level0_h , level0_w), dtype=np.float32)
    tpit_prob_map = np.zeros((level0_h, level0_w), dtype=np.float32)
    patch_size_level0 = 512

    for node_id, node_prob in enumerate(linear_probs_full):
        node_patch_indices = cluster_indices[node_id]
        node_patch_pos = pos_array[node_patch_indices]

        prob_np = node_prob.cpu().numpy() if hasattr(node_prob, 'cpu') else np.array(node_prob)

        for pos in node_patch_pos:
            x_level0_col, y_level0_row = pos[0], pos[1]
            row_start = int(y_level0_row)
            row_end = int(y_level0_row + patch_size_level0)
            col_start = int(x_level0_col)
            col_end = int(x_level0_col + patch_size_level0)

            sf1_prob_map[row_start:row_end, col_start:col_end] = prob_np[0]
            pit1_prob_map[row_start:row_end, col_start:col_end] = prob_np[1]
            tpit_prob_map[row_start:row_end, col_start:col_end] = prob_np[2]

    sf1_prob_resized = cv2.resize(sf1_prob_map, (level3_w, level3_h), interpolation=cv2.INTER_AREA)
    pit1_prob_resized = cv2.resize(pit1_prob_map, (level3_w, level3_h), interpolation=cv2.INTER_AREA)
    tpit_prob_resized = cv2.resize(tpit_prob_map, (level3_w, level3_h), interpolation=cv2.INTER_AREA)

    sf1_color = np.array([89, 24, 126], dtype=np.float32)
    pit1_color = np.array([247, 221, 47], dtype=np.float32)
    tpit_color = np.array([103, 169, 204], dtype=np.float32)
    white = np.array([255, 255, 255], dtype=np.float32)

    def prob_to_color_map(prob_map, target_color):
        h, w = prob_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            color_map[:, :, c] = (white[c] * (1 - prob_map) + target_color[c] * prob_map).astype(np.uint8)
        return color_map

    sf1_color_map = prob_to_color_map(sf1_prob_resized, sf1_color)
    pit1_color_map = prob_to_color_map(pit1_prob_resized, pit1_color)
    tpit_color_map = prob_to_color_map(tpit_prob_resized, tpit_color)

    Image.fromarray(sf1_color_map).save(os.path.join(results_fold_pth, f"{slide_id[0]}_{slide_id[1]}_SF1_prob_map.png"))
    Image.fromarray(pit1_color_map).save(os.path.join(results_fold_pth, f"{slide_id[0]}_{slide_id[1]}_PIT1_prob_map.png"))
    Image.fromarray(tpit_color_map).save(os.path.join(results_fold_pth, f"{slide_id[0]}_{slide_id[1]}_TPIT_prob_map.png"))

    return True










