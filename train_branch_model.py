import os
import re
import torch
import numpy as np
import pandas as pd
import argparse
import h5py
import pickle
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import logging
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import KFold
from tqdm import tqdm
import matplotlib.pylab as plt
from utils.dataset_utils import WSIGraphDataset
from models.GCN import GCN
from skimage.filters  import threshold_otsu
from models.Linear_clf import LinearClassifier
from sklearn.preprocessing  import StandardScaler
from torch_geometric.loader import DataListLoader as PyG_DataLoader
from torch_geometric.nn import DataParallel
from collections import Counter
from scipy.spatial import ConvexHull
from matplotlib.colors import ListedColormap
import umap

device_ids = [0, 1, 2, 3]

print(f"Using device: {device_ids}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_logger(logger_name,root_pth):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(root_pth,f"{logger_name}.log"), mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def extract_scan_info(filename):
    match = re.match(r'^(\d+)(?:-(\d+))?', filename)
    if match:
        scan_number = int(match.group(1))
        subscan_number = int(match.group(2)) if match.group(2) else 0
        return (scan_number, subscan_number)
    else:
        return None

def calculate_quartiles(pair_counts):
    counts = sorted(pair_counts.values())

    q1 = np.percentile(counts, 25, interpolation='linear')
    median = np.percentile(counts, 50, interpolation='linear')
    q3 = np.percentile(counts, 75, interpolation='linear')

    return q1, median, q3

def data_load(args):

    train_cache_path = os.path.join(args.train_test_cache_pth,'train_data_cache.pkl')
    test_cache_path = os.path.join(args.train_test_cache_pth,'test_data_cache.pkl')
    logger = create_logger('load_process',args.train_test_cache_pth)

    if os.path.exists(train_cache_path):
        with open(train_cache_path, 'rb') as f:
            train_dataset = pickle.load(f)
        logger.info(f"load processed pkl training data from {train_cache_path}.")
    if os.path.exists(test_cache_path):
        with open(test_cache_path, 'rb') as f:
            test_dataset = pickle.load(f)
        return train_dataset, test_dataset
    else:
        logger.info(f"no processed cache")
        logger.info(f"loading h5py: {os.path.abspath(args.patch_emb_path)}")
        logger.info(f"loading label: {os.path.abspath(args.label_path)}")

        h5_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(args.patch_emb_path)
            for f in files
            if f.endswith(('.h5', '.hdf5'))
        ]

        used_categories = ["SF-1", "PIT-1", "T-PIT"]
        label_file = pd.read_excel(args.label_path, engine='openpyxl')
        filtered_label = label_file[label_file["分类"].isin(used_categories)]

        train_dataset = {
            "SF-1": dict(),
            "PIT-1": dict(),
            "T-PIT": dict()
        }
        test_dataset = {
            "SF-1": dict(),
            "PIT-1": dict(),
            "T-PIT": dict()
        }

        for cat in used_categories:

            logger.info(f"extract {cat}")

            cat_label_pd = filtered_label[filtered_label["分类"] == cat]
            cat_scans_all = list(cat_label_pd["扫描"])

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

            unique_pairs = np.unique(np.hstack((sub_id_cat, scan_id_cat)), axis=0)
            cat_train, cat_test = train_test_split(unique_pairs, test_size=0.2, shuffle=True)
            logger.info(f"train-{len(cat_train)}: {cat_train}, test-{len(cat_test)}: {cat_test}")
            train_indices = np.isin(np.hstack((sub_id_cat, scan_id_cat)), cat_train).all(axis=1)
            test_indices = np.isin(np.hstack((sub_id_cat, scan_id_cat)), cat_test).all(axis=1)

            train_dataset[cat]["features"] = features_cat[train_indices]
            train_dataset[cat]["corrds"] = coords_cat[train_indices]
            train_dataset[cat]["major_id"] = scan_id_cat[train_indices]
            train_dataset[cat]["sub_id"] = sub_id_cat[train_indices]

            test_dataset[cat]["features"] = features_cat[test_indices]
            test_dataset[cat]["corrds"] = coords_cat[test_indices]
            test_dataset[cat]["major_id"] = scan_id_cat[test_indices]
            test_dataset[cat]["sub_id"] = sub_id_cat[test_indices]

        logger.info(f"saving {train_cache_path}")
        with open(train_cache_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        logger.info(f"saving {test_cache_path}")
        with open(test_cache_path, 'wb') as f:
            pickle.dump(test_dataset, f)

        return train_dataset, test_dataset

def prepare_data_for_cv(dataset, args):
    label_map = {"SF-1": 0, "PIT-1": 1, "T-PIT": 2}

    all_pairs, all_cat_indices = [], []
    for cat in dataset:
        if dataset[cat]["features"] is not None:
            sub_ids = dataset[cat]["sub_id"].reshape(-1, 1)
            scan_ids = dataset[cat]["major_id"].reshape(-1, 1)
            pairs = np.hstack((sub_ids, scan_ids))
            all_pairs.append(pairs)
            all_cat_indices.append(cat)
    all_pairs = np.vstack(all_pairs)
    unique_pairs = np.unique(all_pairs, axis=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(unique_pairs))

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        train_pairs = unique_pairs[train_idx]
        val_pairs = unique_pairs[val_idx]

        fold_input = {k: [] for k in ['features', 'labels', 'major_id', 'sub_id', 'corrds']}
        fold_output = {k: [] for k in ['features', 'labels', 'major_id', 'sub_id', 'corrds']}

        for cat in all_cat_indices:
            data = dataset[cat]
            sub_ids = data["sub_id"].reshape(-1, 1)
            scan_ids = data["major_id"].reshape(-1, 1)
            cat_pairs = np.hstack((sub_ids, scan_ids))

            train_flags = np.isin(cat_pairs, train_pairs).all(axis=1)
            val_flags = np.isin(cat_pairs, val_pairs).all(axis=1)

            fold_input['features'].append(data["features"][train_flags])
            fold_input['labels'].append(np.full(train_flags.sum(), label_map[cat]))
            fold_input['major_id'].append(scan_ids[train_flags])
            fold_input['sub_id'].append(sub_ids[train_flags])
            fold_input['corrds'].append(data["corrds"][train_flags])

            fold_output['features'].append(data["features"][val_flags])
            fold_output['labels'].append(np.full(val_flags.sum(), label_map[cat]))
            fold_output['major_id'].append(scan_ids[val_flags])
            fold_output['sub_id'].append(sub_ids[val_flags])
            fold_output['corrds'].append(data["corrds"][val_flags])

        for key in fold_input:
            fold_input[key] = np.concatenate(fold_input[key])
            fold_output[key] = np.concatenate(fold_output[key])

        with open(os.path.join(args.cross_validation_cache_pth, f'fold_{fold_id + 1}_train_cache.pkl'), 'wb') as f:
            pickle.dump(fold_input, f)
        with open(os.path.join(args.cross_validation_cache_pth, f'fold_{fold_id + 1}_val_cache.pkl'), 'wb') as f:
            pickle.dump(fold_output, f)

def evaluate_model(model, data_loader, criterion=None):

    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_list in data_loader:
            inputs = [item[0] for item in batch_list]
            labels = torch.tensor([item[1] for item in batch_list]).cuda(device=device_ids[0]).squeeze()

            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if criterion is not None:
        total_loss /= len(data_loader)
    else:
        total_loss = 0.0

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss, balanced_acc, macro_f1

def train_model(model, train_loader, val_loader, logger, fold_id, args, num_epochs=50, initial_lr=0.001):

    criterion = FocalLoss(
        alpha=torch.tensor([0.5, 1.0, 1.2], device=device_ids[0]), #1.0 1.5 1.5
        gamma=5.0,
        reduction='mean'
    )

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3, verbose=True)

    best_val_f1 = 0.0
    best_train_f1 = 0.0
    best_model_state = model.state_dict()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    lrs = []

    for epoch in tqdm(range(num_epochs), desc='epoch'):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for batch_list in train_loader:

            inputs = [item[0] for item in batch_list]
            labels = torch.tensor([item[1] for item in batch_list]).to(f'cuda:{device_ids[0]}')
            slide_ids = [item[2] for item in batch_list]
            cluster_indices = [item[3] for item in batch_list]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = balanced_accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion)

        scheduler.step(val_f1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        lrs.append(optimizer.param_groups[0]['lr'])

        logger.info(f'Epoch  {epoch + 1}/{num_epochs}: '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} | '
                    f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            logger.info(f'New  best model with F1: {best_val_f1:.4f}')

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title(f'Fold  {fold_id} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    plt.title(f'Fold  {fold_id} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, 'b-', label='Train F1')
    plt.plot(epochs, val_f1s, 'r-', label='Val F1')
    plt.title(f'Fold  {fold_id} - Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1  Score')
    plt.legend()
    plt.ylim(0,1)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, lrs, 'g-', label='Learning Rate')
    plt.title(f'Fold  {fold_id} - Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning  Rate')
    plt.legend()

    plt.tight_layout()

    plots_dir = os.path.join(args.work_dir, "training_plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'fold_{fold_id}_metrics.png')
    plt.savefig(plot_path)
    plt.close()

    logger.info(f'Saved  training metrics plot for fold {fold_id} to {plot_path}')

    model.load_state_dict(best_model_state)
    return model


def select_patches(atten_scores, all_patch_feats, cluster_indices,labels, pred_labels, pred_probs, all_patch_coords):
    i = 0
    sel_inputs = []
    sel_labels = []
    num_slides = len(labels)
    patches_per_slide = 100

    for i in range(num_slides):
        start_idx = i * patches_per_slide
        end_idx = (i + 1) * patches_per_slide
        if end_idx > len(atten_scores):
            end_idx = len(atten_scores)

        slide_scores = atten_scores[start_idx:end_idx]
        patches_feats = all_patch_feats[i]
        patch_coords = all_patch_coords[i]
        patches_labels = labels[i]
        pred_label = pred_labels[i]
        pred_prob = pred_probs[i]
        if patches_labels.cpu().numpy() != pred_label.cpu().numpy() or pred_prob.cpu().numpy() <= 0.8:
            continue
        cluster_indice = cluster_indices[i]

        k = min(1, slide_scores.shape[0])
        if k == 0:
            continue

        topk_vals, topk_indices = torch.topk(slide_scores, k=k) # topk_indices
        for idx in topk_indices.cpu().tolist():
            used_coords = patch_coords[cluster_indice.get(idx)]

            sel_inputs.append(torch.tensor(patches_feats[cluster_indice.get(idx)],dtype=torch.float32))
            sel_labels.append(torch.tensor([patches_labels] * len(cluster_indice.get(idx)),dtype=torch.long))

    linear_inputs = torch.cat(sel_inputs, dim=0).to('cuda:1')
    linear_labels = torch.cat(sel_labels, dim=0).to('cuda:1')

    return linear_inputs, linear_labels

def _save_umap_plot(embeddings, labels, save_path, colors):

    custom_cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=custom_cmap, s=5, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def evaluate_assistant_model(GCNmodel, Linearmodel, val_loader, criterion=None):

    GCNmodel.eval()
    Linearmodel.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_list in val_loader:
            inputs = [item[0] for item in batch_list]
            labels = torch.tensor([item[1] for item in batch_list]).to(f'cuda:{device_ids[0]}')
            cluster_indices = [item[3] for item in batch_list]
            all_patch_feats = [item[5] for item in batch_list]
            all_patch_coords = [item[4] for item in batch_list]

            gcn_outputs = GCNmodel(inputs)
            pred_probs, pred_labels = torch.max(F.softmax(gcn_outputs, dim=-1), 1)
            atten_scores = GCNmodel.module.node_importance_scores

            linear_inputs, linear_labels = select_patches(
                atten_scores, all_patch_feats, cluster_indices,
                labels, pred_labels, pred_probs, all_patch_coords,
            )
            linear_outputs = Linearmodel(linear_inputs)
            if criterion is not None:
                loss = criterion(linear_outputs, linear_labels)
                total_loss += loss.item()
            _, preds = torch.max(linear_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(linear_labels.cpu().numpy())

    if criterion is not None:
        total_loss /= len(val_loader)

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss, balanced_acc, macro_f1


def train_assistant_model(GCNmodel, Linearmodel, train_loader, val_loader, fold_id, args, num_epochs=50, initial_lr=0.001):

    logger = create_logger('5CV_assistant_traning', root_pth=args.work_dir)
    criterion = FocalLoss(
        alpha=torch.tensor([0.5, 1.0, 1.0], device='cuda:1'),
        gamma=6.0,
        reduction='mean',
    )
    optimizer = optim.AdamW(Linearmodel.parameters(), lr=initial_lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=2, verbose=True)

    best_val_f1 = 0.0
    best_model_state = Linearmodel.state_dict()
    early_stopping_cnts = 0
    max_cnt = 5

    GCNmodel.eval()

    for epoch in tqdm(range(num_epochs), desc='epoch'):
        Linearmodel.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for batch_list in train_loader:
            graph_inputs = [item[0] for item in batch_list]
            labels = torch.tensor([item[1] for item in batch_list]).to('cuda:1')
            cluster_indices = [item[3] for item in batch_list]
            all_patch_feats = [item[5] for item in batch_list]
            all_patch_coords = [item[4] for item in batch_list]

            with torch.no_grad():
                gcn_outputs = GCNmodel(graph_inputs)
                pred_probs, pred_labels = torch.max(F.softmax(gcn_outputs, dim=-1), 1)
                atten_scores = GCNmodel.module.node_importance_scores

            linear_inputs, linear_labels = select_patches(
                atten_scores, all_patch_feats, cluster_indices,
                labels, pred_labels, pred_probs, all_patch_coords,
            )

            perm = torch.randperm(linear_inputs.size(0))
            linear_inputs = linear_inputs[perm]
            linear_labels = linear_labels[perm]

            optimizer.zero_grad()
            linear_outputs = Linearmodel(linear_inputs.to('cuda:1'))
            loss = criterion(linear_outputs.to('cuda:1'), linear_labels.to('cuda:1'))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(linear_outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(linear_labels.cpu().numpy())

        train_loss /= max(1, len(train_loader))
        train_acc = balanced_accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

        val_loss, val_acc, val_f1 = evaluate_assistant_model(GCNmodel, Linearmodel, val_loader, criterion)
        scheduler.step(val_f1)

        logger.info(
            f'Epoch {epoch + 1}/{num_epochs}: '
            f'[Linear] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | '
            f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_model_state = Linearmodel.state_dict()
            logger.info(f'[Linear] New best model with F1: {best_val_f1:.4f}')
            early_stopping_cnts = 0
        else:
            early_stopping_cnts += 1
            if early_stopping_cnts == max_cnt:
                break

    model_save_pth = os.path.join(args.work_dir, "saved_models")
    os.makedirs(model_save_pth, exist_ok=True)
    linear_model_pth = os.path.join(model_save_pth, f'best_model_fold_{fold_id}_assistant.pth')
    torch.save(best_model_state, linear_model_pth)
    logger.info(f'Saved best assistant model for fold {fold_id} to {linear_model_pth}')


def visualize_umap_per_fold(GCNmodel, train_loader, fold_id, args):

    assistant_model_pth = os.path.join(args.work_dir, 'saved_models', f'best_model_fold_{fold_id}_assistant.pth')
    if not os.path.exists(assistant_model_pth):
        raise FileNotFoundError(
            f'Assistant model weights not found: {assistant_model_pth}. '
            f'Run train_assistant_model first.'
        )

    assistant_model = LinearClassifier(input_dim=1536, output_dim=3).to('cuda:1')
    assistant_model.load_state_dict(torch.load(assistant_model_pth, weights_only=True))
    assistant_model.eval()
    GCNmodel.eval()

    UMAP_feats_ls = []
    UMAP_feats_learned_ls = []
    UMAP_labels_ls = []

    for batch_list in train_loader:
        graph_inputs = [item[0] for item in batch_list]
        labels = torch.tensor([item[1] for item in batch_list]).to('cuda:1')
        cluster_indices = [item[3] for item in batch_list]
        all_patch_feats = [item[5] for item in batch_list]
        all_patch_coords = [item[4] for item in batch_list]

        with torch.no_grad():
            gcn_outputs = GCNmodel(graph_inputs)
            pred_probs, pred_labels = torch.max(F.softmax(gcn_outputs, dim=-1), 1)
            atten_scores = GCNmodel.module.node_importance_scores

        linear_inputs, linear_labels = select_patches(
            atten_scores, all_patch_feats, cluster_indices,
            labels, pred_labels, pred_probs, all_patch_coords,
        )

        UMAP_feats_ls.append(linear_inputs.cpu().numpy())
        with torch.no_grad():
            UMAP_feats_learned_ls.append(assistant_model(linear_inputs).cpu().detach().numpy())
        UMAP_labels_ls.append(linear_labels.cpu().numpy())

    UMAP_feats = np.vstack(UMAP_feats_ls)
    UMAP_feats_learned = np.vstack(UMAP_feats_learned_ls)
    UMAP_labels = np.concatenate(UMAP_labels_ls)

    custom_colors = [
        (89.0 / 255.0, 24.0 / 255.0, 126.0 / 255.0),
        (247.0 / 255.0, 221.0 / 255.0, 47.0 / 255.0),
        (103.0 / 255.0, 169.0 / 255.0, 204.0 / 255.0),
    ]
    umap_kwargs = dict(
        n_components=2,
        n_neighbors=100,
        min_dist=0.1,
        spread=1.5,
        repulsion_strength=0.5,
        local_connectivity=2,
        target_metric='cosine',
        random_state=42,
    )

    reducer = umap.UMAP(**umap_kwargs)
    umap_embeddings_unlearned = reducer.fit_transform(UMAP_feats)
    _save_umap_plot(
        umap_embeddings_unlearned,
        UMAP_labels,
        os.path.join(args.work_dir, f'UMAP_fold_{fold_id}_unlearned.tiff'),
        custom_colors,
    )

    reducer_learned = umap.UMAP(**umap_kwargs)
    umap_embeddings_learned = reducer_learned.fit_transform(UMAP_feats_learned)
    _save_umap_plot(
        umap_embeddings_learned,
        UMAP_labels,
        os.path.join(args.work_dir, f'UMAP_fold_{fold_id}_learned.tiff'),
        custom_colors,
    )

def stratified_kfold_cv(dataset, n_splits, batch_size, args):

    model_save_pth = os.path.join(args.work_dir, "saved_models")
    tensorboard_logs_pth = os.path.join(args.work_dir, "tensorboard_train_logs")
    os.makedirs(model_save_pth, exist_ok=True)
    os.makedirs(tensorboard_logs_pth, exist_ok=True)
    logger = create_logger('5CV_traing', root_pth=args.work_dir)

    if not os.path.exists(args.cross_validation_cache_pth):
        os.makedirs(args.cross_validation_cache_pth)
        prepare_data_for_cv(dataset, args)

    for fold_id in [5]:

        logger.info(f"Fold  {fold_id} start")
        with open(os.path.join(args.cross_validation_cache_pth, f'fold_{fold_id}_train_cache.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(args.cross_validation_cache_pth, f'fold_{fold_id}_val_cache.pkl'), 'rb') as f:
            val_data = pickle.load(f)

        logger.info(f"train  patch samplesize : {train_data['features'].shape[0]}")
        logger.info(f"val  patch samplesize : {val_data['features'].shape[0]}")
        logger.info(
            f'train  slide samplesize: {len(np.unique(np.hstack((train_data["major_id"], train_data["sub_id"])), axis=0))}')
        logger.info(
            f'val  slide samplesize: {len(np.unique(np.hstack((val_data["major_id"], val_data["sub_id"])), axis=0))}')

        traindata = WSIGraphDataset(patch_embed=train_data['features'],
                                    patch_label=train_data['labels'],
                                    patch_majorid=train_data['major_id'],
                                    patch_subid=train_data['sub_id'],
                                    patch_coords=train_data['corrds'])

        valdata = WSIGraphDataset(patch_embed=val_data['features'],
                                    patch_label=val_data['labels'],
                                    patch_majorid=val_data['major_id'],
                                    patch_subid=val_data['sub_id'],
                                    patch_coords=val_data['corrds'])

        trainloader = PyG_DataLoader(
            traindata,
            batch_size=batch_size * len(device_ids),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        valloader = PyG_DataLoader(
            valdata,
            batch_size=batch_size * len(device_ids),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        if not os.path.exists(os.path.join(args.work_dir,'saved_models',f'best_model_fold_{fold_id}.pth')):

            GCNmodel = GCN(input_dim=1536, hidden_dim=512, num_classes=3)

            GCNmodel = DataParallel(GCNmodel, device_ids=device_ids)
            GCNmodel = GCNmodel.cuda(device=device_ids[0])
            best_GCNmodel = train_model(GCNmodel, trainloader, valloader, logger, fold_id, args, num_epochs=8, initial_lr=0.001)
            val_loss, val_acc, val_f1 = evaluate_model(best_GCNmodel, valloader)
            logger.info(f'fold_{fold_id}:  val_macro-f1={val_f1:.4f}, val_balanced_acc={val_acc:.4f}')
            model_path = os.path.join(model_save_pth, f'best_model_fold_{fold_id}.pth')
            torch.save(best_GCNmodel.state_dict(), model_path)
            logger.info(f'Saved  best model for fold {fold_id} to {model_path}')
        else:
            GCN_pth = os.path.join(args.work_dir,'saved_models',f'best_model_fold_{fold_id}.pth')
            GCNmodel = GCN(input_dim=1536, hidden_dim=512, num_classes=3)
            GCNmodel = DataParallel(GCNmodel, device_ids=[0])
            GCNmodel.load_state_dict(torch.load(GCN_pth, weights_only=True))
            GCNmodel = GCNmodel.cuda(device="cuda:0")
            GCNmodel.eval()


        Linearmodel = LinearClassifier(input_dim=1536, output_dim=3).to("cuda:1")
        train_assistant_model(GCNmodel, Linearmodel, trainloader, valloader, fold_id, args, num_epochs=20, initial_lr=0.001)
        visualize_umap_per_fold(GCNmodel, trainloader, fold_id, args)


def dataset_describe(args, dataset, dataset_type):

    all_pairs = []
    for type_, infos in dataset.items():
        major_ids = infos['major_id']
        sub_ids = infos['sub_id']
        pairs = []
        for j in range(0, len(major_ids)):
            pairs.append((str(major_ids[j][0]), str(sub_ids[j][0])))
            all_pairs.append((str(major_ids[j][0]), str(sub_ids[j][0])))
        unique_pairs = list(set(pairs))

        describe_txt_pth = os.path.join(args.work_dir, f"{dataset_type}_describe_{type_}.txt")
        with open(describe_txt_pth, 'w') as f:
            f.write(f'{type_}\n')
            f.write(f'slides num:{len(unique_pairs)}\n')
            f.write(f'patches num:{len(pairs)}\n')
            f.write(str(unique_pairs))
            f.close()

    pair_counts = Counter(all_pairs)
    q1, median, q3 = calculate_quartiles(pair_counts)
    return q1, median, q3

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PitNET branch-model (LinearClassifier) training, reuses a main-model run directory')
    parser.add_argument('--patch_emb_path', type=str, default='./data/patch_features',
                        help='Directory of pre-computed patch features (.h5)')
    parser.add_argument('--label_path', type=str, default='./data/labels.xlsx',
                        help='Excel file with slide-level labels (columns: id, subtype)')
    parser.add_argument('--feat_dim', type=int, default=1536)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--magnitude', type=int, default=40)
    parser.add_argument('--data_source', type=str, default='TianTan')
    parser.add_argument('--inter_or_outer', type=str, default='internal')
    parser.add_argument('--task_phase', type=str, default='UMAP_vis')
    parser.add_argument('--emb_model', type=str, default='uni2')
    parser.add_argument('--task', type=str, default='pitnet')
    parser.add_argument('--train_test_cache_pth', type=str, default='./cache/train_test',
                        help='Cache directory for serialized train/test splits')
    parser.add_argument('--cross_validation_cache_pth', type=str, default='./cache/cv',
                        help='Cache directory for cross-validation folds')
    parser.add_argument('--main_work_dir', type=str, required=True,
                        help='Working directory of a completed main-model run (contains saved_models/best_model_fold_*.pth). '
                             'The branch model is trained on top and writes best_model_fold_*_assistant.pth into the same saved_models/')
    args = parser.parse_args()

    current_work_dir = args.main_work_dir
    os.makedirs(current_work_dir, exist_ok=True)
    setattr(args, 'work_dir', current_work_dir )

    if not os.path.exists(args.train_test_cache_pth):
        os.makedirs(args.train_test_cache_pth)
        train_dataset, test_dataset = data_load(args)
    else:
        with open(os.path.join(args.train_test_cache_pth,'train_data_cache.pkl'), 'rb') as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(args.train_test_cache_pth,'test_data_cache.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)

    train_q1, train_median, train_q3 = dataset_describe(args, train_dataset, 'traindataset')
    test_q1, test_median, test_q3 = dataset_describe(args, test_dataset, 'testdataset')

    stratified_kfold_cv(dataset = train_dataset, n_splits = 5, batch_size = 8, args = args)
