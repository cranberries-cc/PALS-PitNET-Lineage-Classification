from datetime import datetime
import torch
import os
import numpy as np
import pandas as pd
import argparse
import h5py
from tqdm import tqdm
from pathlib import Path
import pickle
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from utils.dataset_utils import WSIGraphDataset
from utils.file_utils import *
from utils.interpretability_utils import *
from sklearn.preprocessing  import StandardScaler
from openpyxl import Workbook
import torch.nn.functional as F
from torch_geometric.loader import DataListLoader as PyG_DataLoader
from torch_geometric.nn import DataParallel
from models.GCN import GCN
from models.Linear_clf import LinearClassifier
from skimage.filters  import threshold_otsu

def data_load_external_validation(external_emb_pth):
    h5_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(external_emb_pth)
        for f in files
        if f.endswith(('.h5', '.hdf5'))
    ]
    external_dataset = {
        "多发同步不同谱系" : dict()
    }

    coords_cat = None
    features_cat = None
    sub_id_cat = None
    scan_id_cat = None

    for h5_file in h5_files:
        scan_id, sub_id = extract_scan_info(h5_file.split("/")[-1])

        with h5py.File(h5_file, 'r') as f:
            coords = f['coords'][()]
            features = f['features'][()]

            patch_num = coords.shape[0]

            sub_id_arr = np.full((patch_num, 1), sub_id)
            scan_id_arr = np.full((patch_num, 1), scan_id)

            if coords_cat is None:
                coords_cat = coords
                features_cat = features
                sub_id_cat = sub_id_arr
                scan_id_cat = scan_id_arr
            else:
                coords_cat = np.vstack((coords_cat, coords))
                features_cat = np.vstack((features_cat, features))
                sub_id_cat = np.vstack((sub_id_cat, sub_id_arr))
                scan_id_cat = np.vstack((scan_id_cat, scan_id_arr))

    external_dataset["多发同步不同谱系"]["features"] = features_cat
    external_dataset["多发同步不同谱系"]["corrds"] = coords_cat
    external_dataset["多发同步不同谱系"]["major_id"] = scan_id_cat
    external_dataset["多发同步不同谱系"]["sub_id"] = sub_id_cat

    return external_dataset

def select_patches(atten_scores, inputs, labels):
    i = 0
    sel_inputs = []
    sel_labels = []
    sel_indices = []
    num_slides = len(labels)
    patches_per_slide = 100

    for i in range(num_slides):
        start_idx = i * patches_per_slide
        end_idx = (i + 1) * patches_per_slide
        if end_idx > len(atten_scores):
            end_idx = len(atten_scores)

        slide_scores = atten_scores[start_idx:end_idx]
        slide_inputs = inputs[i].x
        slide_label = labels[i]
        scores_np = slide_scores.cpu().numpy()

        if np.all(scores_np == scores_np[0]):
            otsu_thresh = np.mean(scores_np)
        else:
            otsu_thresh = threshold_otsu(scores_np)

        selection_mask = scores_np >= otsu_thresh

        for j, selected in enumerate(selection_mask):
            if selected:
                sel_indices.append(j)
                sel_inputs.append(slide_inputs[j])
                sel_labels.append(slide_label)

    linear_inputs = torch.stack(sel_inputs).float().to('cuda:1')
    linear_labels = torch.tensor(sel_labels).long().to('cuda:1')

    return linear_inputs, linear_labels, sel_indices

def slide_level_analysis(GCNmodel, assistant_model, testloader, results_fold_pth, args):

    max_vis_count = 10
    vis_count = 0
    slide_pre_all = []
    slide_gt_all = []

    fold_predict_info_wkb = Workbook()
    fold_predict_info_wks = fold_predict_info_wkb.active
    fold_predict_info_wks.title = f'fold{fold_id}_slide_prediction_info'
    fold_predict_info_header = ["患者id", "子扫描id", "预测SF1概率", "预测PIT1概率", "预测TPIT概率", "最大概率预测标签", "真实标签"]
    fold_predict_info_wks.append(fold_predict_info_header)

    with torch.no_grad():

        for batch_list in testloader:
            inputs = [item[0] for item in batch_list]
            single_input = inputs[0]
            single_input = single_input.to(f'cuda:{device_ids[0]}')
            labels = torch.tensor([item[1] for item in batch_list]).to(f'cuda:{device_ids[0]}')
            slide_id = [item[2] for item in batch_list]
            cluster_indices = [item[3] for item in batch_list]
            coords = [item[4] for item in batch_list]

            _ = GCNmodel.module(single_input)
            atten_scores = GCNmodel.module.node_importance_scores

            linear_inputs_partial, linear_labels, selected_indices = select_patches(atten_scores, inputs, labels)
            linear_outputs_partial = assistant_model(linear_inputs_partial.to(f'cuda:0'))
            __, linear_preds_partial = torch.max(linear_outputs_partial, 1)

            linear_inputs_full = inputs[0].x
            linear_outputs_full = assistant_model(linear_inputs_full.to(f'cuda:0'))
            linear_probs_full = F.softmax(linear_outputs_full)
            ___, linear_preds_full = torch.max(linear_outputs_full, 1)

            visualize_Multi_hormone(args, slide_id[0], results_fold_pth, coords[0], cluster_indices[0],
                                    linear_preds_full, linear_probs_full, linear_preds_partial, selected_indices,
                                    tresh=0.6)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PitNET branch-model (LinearClassifier) evaluation, multi-hormone / multi-synchronous mode')
    parser.add_argument('--models_pths', type=str, default='./saved_models',
                        help='Directory containing both main-model (best_model_fold_*.pth) and branch-model (best_model_fold_*_assistant.pth) weights')
    parser.add_argument('--wsi_pth', type=str, default='./data/wsi',
                        help='Directory of original whole-slide images (used for overlay visualisation)')
    parser.add_argument('--feat_dim', type=int, default=1536, help='Dim of patch embedding feature vector')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of pituitary lineages')
    parser.add_argument('--mag', type=int, default=40, help='Magnification')
    parser.add_argument('--data_source', type=str, default='TianTan', help='Source institution of the data')
    parser.add_argument('--inter_or_outer', type=str, default='internal', help='internal / inter / outer cohort')
    parser.add_argument('--task_phase', type=str, default='branch_model', help='Tag used in the output directory name')
    parser.add_argument('--emb_model', type=str, default='uni2', help='Patch embedding backbone')
    parser.add_argument('--task', type=str, default='pitnet', help='Task name (used in output directory)')
    parser.add_argument('--data_processed_cache_pth', type=str, default=None,
                        help='Optional pickled cache of a pre-built test dataset')
    parser.add_argument('--external_emb_pth', type=str, default='./data/external_features',
                        help='Directory of pre-computed patch features for external validation / multi-synchronous cohort')
    parser.add_argument('--output_root', type=str, default='./outputs',
                        help='Root directory for evaluation outputs')
    args = parser.parse_args()

    device_ids = [0]
    label_map = {"多发同步不同谱系": -1}
    class_name = {-1: "多发同步不同谱系"}
    color_map = {"多发同步不同谱系": 'y'}

    results_root = os.path.join(
        args.output_root,
        f'{args.task}_{args.data_source}_{args.inter_or_outer}_{args.task_phase}_{args.emb_model}'
        f'_mag{args.mag}_feat{args.feat_dim}_class{args.num_classes}'
        f'_{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    )
    os.makedirs(results_root, exist_ok=True)

    dataset = data_load_external_validation(external_emb_pth=args.external_emb_pth)

    test_dataset = dataset_convert(dataset, label_map)

    testdata =  WSIGraphDataset(patch_embed=test_dataset['features'],
                                    patch_label=test_dataset['labels'],
                                    patch_majorid=test_dataset['major_id'],
                                    patch_subid=test_dataset['sub_id'],
                                    patch_coords=test_dataset['corrds'])
    testloader = PyG_DataLoader(
            testdata,
            batch_size= 1 * len(device_ids),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    for fold_id in [0,1,2,3,4,5]:

        print(f'[fold: {fold_id}]')
        results_fold_pth = os.path.join(results_root, f'flod_{fold_id}')
        os.makedirs(results_fold_pth, exist_ok=True)

        gcn_model_pth = os.path.join(args.models_pths, f'best_model_fold_{fold_id}.pth')
        GCNmodel = GCN(input_dim=1536, hidden_dim=512, num_classes=3)
        GCNmodel = DataParallel(GCNmodel, device_ids=[0])
        GCNmodel.load_state_dict(torch.load(gcn_model_pth,weights_only=True))
        GCNmodel = GCNmodel.cuda(device=0)
        GCNmodel.eval()

        assistant_model_pth = os.path.join(args.models_pths, f'best_model_fold_{fold_id}_assistant.pth')
        assistant_model = LinearClassifier(input_dim=1536,output_dim=3).to("cuda:0")
        assistant_model.load_state_dict(torch.load(assistant_model_pth,weights_only=True))
        assistant_model.eval()

        slide_level_analysis(GCNmodel, assistant_model, testloader, results_fold_pth, args)

