import re
import os
import numpy as np
import openslide
from openslide import OpenSlide

def extract_scan_info(filename):
    match = re.match(r'^(\d+)(?:-(\d+))?',  filename)
    if match:
        scan_number = int(match.group(1))
        subscan_number = int(match.group(2))  if match.group(2)  else 0
        return (scan_number, subscan_number)
    else:
        return None

def dataset_convert(dataset, label_map):

    new_dict = {'features':[], 'labels':[], 'major_id':[], 'sub_id':[], 'corrds':[]}
    X_list = []
    y_list = []
    scan_id_list = []
    sub_id_list = []
    coords_list = []

    for cat in dataset.keys():
        features = dataset[cat]["features"]
        scan_ids = dataset[cat]["major_id"].flatten()
        sub_ids = dataset[cat]["sub_id"].flatten()
        cord = dataset[cat]["corrds"]
        labels = np.full((features.shape[0],), label_map[cat])

        X_list.append(features)
        y_list.append(labels)
        scan_id_list.append(scan_ids)
        coords_list.append(cord)
        sub_id_list.append(sub_ids)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).reshape(-1, 1)
    scan_ids = np.concatenate(scan_id_list, axis=0).reshape(-1, 1)
    sub_ids = np.concatenate(sub_id_list, axis=0).reshape(-1, 1)
    corrds = np.concatenate(coords_list, axis=0)

    new_dict['features'] = X
    new_dict['labels'] = y
    new_dict['major_id'] =  scan_ids
    new_dict['sub_id'] = sub_ids
    new_dict['corrds'] = corrds

    return new_dict

def find_target_wsi_img(major_id,sub_id,wsi_pth):

    ndpi_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(wsi_pth)
        for f in files
        # if f.endswith('.ndpi') or f.endswith('.svs')
    ]

    for nf in ndpi_files:
        scan_id_file, scan_sub_id_file = extract_scan_info(os.path.basename(nf))
        if scan_id_file == major_id and scan_sub_id_file == sub_id:
            slide_target = OpenSlide(nf)
            return slide_target

    return None

