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
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from models.GCN import GCN
import seaborn as sns
from itertools import cycle
from sklearn.metrics  import recall_score

def data_load_external_validation(external_emb_pth,external_label_pth,results_root, is_mul_sync=False): #专门用于外部验证数据的编码
    h5_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(external_emb_pth)
        for f in files
        if f.endswith(('.h5', '.hdf5'))
    ]

    if not is_mul_sync:

        used_categories = ["SF-1", "PIT-1", "T-PIT"]
        label_file = pd.read_excel(external_label_pth, engine='openpyxl')
        filtered_label = label_file[label_file["subtype"].isin(used_categories)]

        external_dataset = {
            "SF-1": dict(),
            "PIT-1": dict(),
            "T-PIT": dict()
        }

        for cat in used_categories:

            cat_label_pd = filtered_label[filtered_label["subtype"] == cat]
            cat_scans_all = list(cat_label_pd["id"])

            coords_cat = None
            features_cat = None
            sub_id_cat = None
            scan_id_cat = None

            for h5_file in h5_files:
                scan_id, sub_id = extract_scan_info(h5_file.split("/")[-1])

                if scan_id in cat_scans_all:
                    with h5py.File(h5_file, 'r') as f:
                        coords = f['coords'][()]
                        features = f['features'][()]

                    assert coords.shape[0] == features.shape[0]
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

            external_dataset[cat]["features"] = features_cat
            external_dataset[cat]["corrds"] = coords_cat
            external_dataset[cat]["major_id"] = scan_id_cat
            external_dataset[cat]["sub_id"] = sub_id_cat

        with open(os.path.join(results_root,'ext_data_cache.pkl'), 'wb') as f:
            pickle.dump(external_dataset, f)
    else:
        external_dataset = {
            "sync": dict()
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

        external_dataset["sync"]["features"] = features_cat
        external_dataset["sync"]["corrds"] = coords_cat
        external_dataset["sync"]["major_id"] = scan_id_cat
        external_dataset["sync"]["sub_id"] = sub_id_cat

    return external_dataset

def slide_level_analysis(model,testloader,results_fold_pth,args):

    max_vis_count = 40
    vis_count = 0
    slide_pre_all = []
    slide_prob_all = []
    slide_gt_all = []

    fold_predict_info_wkb = Workbook()
    fold_predict_info_wks = fold_predict_info_wkb.active
    fold_predict_info_wks.title = f'fold{fold_id}_slide_prediction_info'
    fold_predict_info_header = ["患者id", "子扫描id", "预测SF1概率", "预测PIT1概率", "预测TPIT概率", "最大概率预测标签", "真实标签"]
    fold_predict_info_wks.append(fold_predict_info_header)

    with torch.no_grad():

        for batch_list in testloader:
            single_graph = [item[0] for item in batch_list][0].to(f'cuda:0')
            gt = torch.tensor([item[1] for item in batch_list]).to(f'cuda:0') # (batchsize, 1)
            slide_id = [item[2] for item in batch_list] # (batchsize ,2)
            cluster_indices = [item[3] for item in batch_list]
            coords = [item[4] for item in batch_list]

            outputs = model(single_graph)
            outputs = outputs[-1].view(1,-1)
            #根据slide_id找到wsi源文件，并绘制overlay

            slide_gt_all.append(gt.cpu().numpy())
            pred_label = torch.max(outputs, dim=1)[1]# (batchsize, 1)
            slide_pre_all.append(pred_label.cpu().numpy())
            pred_prob = F.softmax(outputs, dim=1) # (batchsize, 3)
            slide_prob_all.append(pred_prob.cpu().numpy())
            pred_confi = pred_prob[:,pred_label].squeeze()

            pred_prob_np = pred_prob.cpu().numpy()[0]
            fold_predict_info_wks.append([
                f"{slide_id[0][0]}",
                f"{slide_id[0][1]}",
                f"{pred_prob_np[0]:.4f}",
                f"{pred_prob_np[1]:.4f}",
                f"{pred_prob_np[2]:.4f}",
                int(pred_label.cpu().numpy()[0]),
                int(gt.cpu().numpy()[0])
            ])

    fold_predict_info_wkb.save(f"{results_fold_pth}/detailed_slide_prediction_info.xlsx")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PitNET main-model evaluation (slide-level metrics, confusion matrix, heatmaps)')
    parser.add_argument('--models_pths', type=str, default='./saved_models',
                        help='Directory containing main-model weights best_model_fold_{1..5}.pth')
    parser.add_argument('--wsi_pth', type=str, default='./data/wsi',
                        help='Directory of original whole-slide images (used for overlay visualisation)')
    parser.add_argument('--feat_dim', type=int, default=1536)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--mag', type=int, default=40)
    parser.add_argument('--data_source', type=str, default='TianTan')
    parser.add_argument('--inter_or_outer', type=str, default='inter')
    parser.add_argument('--task_phase', type=str, default='main_model')
    parser.add_argument('--emb_model', type=str, default='uni2')
    parser.add_argument('--task', type=str, default='pitnet')
    parser.add_argument('--data_processed_cache_pth', type=str, default=None,
                        help='Optional pickled cache of a pre-built test dataset; if set, --external_emb_pth/--external_label_pth are ignored')
    parser.add_argument('--external_emb_pth', type=str, default='./data/external_features',
                        help='Directory of pre-computed patch features for the external validation cohort')
    parser.add_argument('--external_label_pth', type=str, default='./data/external_labels.xlsx',
                        help='Excel file with external cohort labels (columns: id, subtype)')
    parser.add_argument('--output_root', type=str, default='./outputs',
                        help='Root directory for evaluation outputs')
    args = parser.parse_args()

    device_ids = [0]
    label_map = {"sync": -1, "SF-1": 0, "PIT-1": 1, "T-PIT": 2}
    class_name = {0: 'SF-1', 1: 'PIT-1', 2: 'T-PIT'}
    color_map = {"SF-1": 'r', "PIT-1": 'g', "T-PIT": 'b'}

    results_root = os.path.join(
        args.output_root,
        f'{args.task}_{args.data_source}_{args.inter_or_outer}_{args.task_phase}_{args.emb_model}'
        f'_mag{args.mag}_feat{args.feat_dim}_class{args.num_classes}'
        f'_{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    )
    os.makedirs(results_root, exist_ok=True)

    dataset = None
    if args.data_processed_cache_pth is not None:
        with open(args.data_processed_cache_pth,'rb') as f:
            print(f'start {args.data_source} {args.inter_or_outer} validation ..')
            dataset = pickle.load(f)
            print(f"loading data from: {args.data_processed_cache_pth}")
    else:
        print(f'prepare for {args.data_source} {args.inter_or_outer} validation ..')
        dataset = data_load_external_validation(external_emb_pth=args.external_emb_pth,
                                                external_label_pth=args.external_label_pth,
                                                results_root=results_root,
                                                is_mul_sync=True)
        print('preparation done, start testing..')

    test_dataset = dataset_convert(dataset, label_map)

    testdata =  WSIGraphDataset(patch_embed=test_dataset['features'],
                                    patch_label=test_dataset['labels'],
                                    patch_majorid=test_dataset['major_id'],
                                    patch_subid=test_dataset['sub_id'],
                                    patch_coords=test_dataset['corrds'])
    testloader = PyG_DataLoader(
            testdata,
            batch_size= 1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    for cv_model_pth in tqdm(Path(args.models_pths).iterdir(),desc='Fold'):

        cv_model_pth = str(cv_model_pth)
        fold_id = os.path.basename(cv_model_pth).split('.')[0].split('_')[-1]
        if fold_id == "assistant" or int(fold_id) not in [5]:
            continue
        print(f'[fold: {fold_id}]')
        results_fold_pth = os.path.join(results_root,f'flod_{fold_id}')
        os.makedirs(results_fold_pth,exist_ok=True)

        fold_model = GCN(input_dim=1536, hidden_dim=512, num_classes=3)
        state_dict = torch.load(cv_model_pth, map_location='cpu', weights_only=True)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        fold_model.load_state_dict(new_state_dict)
        fold_model = fold_model.cuda(device=0)
        fold_model.eval()

        slide_level_analysis(fold_model, testloader, results_fold_pth, args)





















