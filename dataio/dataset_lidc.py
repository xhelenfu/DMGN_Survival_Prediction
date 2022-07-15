import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import tifffile
import imgaug.augmenters as iaa

from .utils import *

class DatasetLIDC(data.Dataset):
    def __init__(self, data_sources, fold_id, fold_mean, fold_std, in_h, in_w,
                 di_ct, di_rt, di_ht, di_er, cont_cols, cat_cols, n_classes_cat,
                 isTraining=True):

        # Check valid data directories
        if not os.path.exists(data_sources.img_data_dir):
            sys.exit("Invalid images directory %s" %data_sources.img_data_dir)
        if not os.path.exists(data_sources.labels_file):
            sys.exit("Invalid feature labels path %s" %data_sources.labels_file)

        self.img_data_dir = data_sources.img_data_dir
        self.fold_splits = data_sources.fold_splits
        self.labels_file = data_sources.labels_file
        self.fold_id = fold_id
        self.fold_mean = fold_mean
        self.fold_std = fold_std
        self.in_height = in_h 
        self.in_width = in_w
        self.isTraining = isTraining

        self.di_ct = di_ct
        self.di_rt = di_rt 
        self.di_ht = di_ht
        self.di_er = di_er
        self.n_classes_cat = n_classes_cat

        # Data files
        if isTraining:
            patient_subset_txt = self.fold_splits + '/' + str(self.fold_id) + '_train.txt'
            patient_subset_norm = self.fold_splits + '/' + str(self.fold_id) + '_train.txt'
        else:
            patient_subset_txt = self.fold_splits + '/' + str(self.fold_id) + '_test.txt'
            patient_subset_norm = self.fold_splits + '/' + str(self.fold_id) + '_train.txt'

        self.img_paths = get_data_dirs_split(patient_subset_txt, self.img_data_dir)

        # Samples for normalising continuous variables
        self.img_paths_norm = get_data_dirs_split(patient_subset_norm, self.img_data_dir)
        self.img_paths_norm = [os.path.basename(x).split('_')[0] for x in self.img_paths_norm]
        self.img_paths_norm = [x[:2] + '-' + x[2:] for x in self.img_paths_norm]

        # All the labels for this data subset
        data_df = pd.read_csv(self.labels_file)
        self.labels_df = data_df.iloc[:,:3]
                
        # Clinical data starts from 4th column
        clinical_df = data_df[cont_cols]
        clinical_df_cat = data_df[cat_cols]
        
        self.clinical_df = clinical_df*1
        self.clinical_df_cat = clinical_df_cat*1
        
        # Normalise continuous vars: Age.At.Diagnosis, Ki67, EGFR, PR, HER2
        norm_df = self.clinical_df[self.labels_df['METABRIC.ID'].isin(self.img_paths_norm)]
        self.clinical_df["Age.At.Diagnosis"] = (clinical_df["Age.At.Diagnosis"]-norm_df["Age.At.Diagnosis"].mean())/norm_df["Age.At.Diagnosis"].std()
        self.clinical_df["Ki67"] = (clinical_df["Ki67"]-norm_df["Ki67"].mean())/norm_df["Ki67"].std()
        self.clinical_df["EGFR"] = (clinical_df["EGFR"]-norm_df["EGFR"].mean())/norm_df["EGFR"].std()
        self.clinical_df["PR"] = (clinical_df["PR"]-norm_df["PR"].mean())/norm_df["PR"].std()
        self.clinical_df["HER2"] = (clinical_df["HER2"]-norm_df["HER2"].mean())/norm_df["HER2"].std()
        
        # Map categorical to one-hot
        self.clinical_df_cat['CT'] = clinical_df_cat['CT'].map(self.di_ct)
        self.clinical_df_cat['RT'] = clinical_df_cat['RT'].map(self.di_rt)
        self.clinical_df_cat['HT'] = clinical_df_cat['HT'].map(self.di_ht)
        self.clinical_df_cat['ER.Status'] = clinical_df_cat['ER.Status'].map(self.di_er)
        

    def augment_data(self, batch_raw):
        batch_raw = np.expand_dims(batch_raw, 0)

        # Original, horizontal
        random_flip = np.random.randint(2, size=1)[0]
        # 0, 90, 180, 270
        random_rotate = np.random.randint(4, size=1)[0]

        # Flips
        if random_flip == 0:
            batch_flip = batch_raw*1
        else:
            batch_flip = iaa.Flipud(1.0)(images=batch_raw)
                
        # Rotations
        if random_rotate == 0:
            batch_rotate = batch_flip*1
        elif random_rotate == 1:
            batch_rotate = iaa.Rot90(1, keep_size=True)(images=batch_flip)
        elif random_rotate == 2:
            batch_rotate = iaa.Rot90(2, keep_size=True)(images=batch_flip)
        else:
            batch_rotate = iaa.Rot90(3, keep_size=True)(images=batch_flip)
        
        images_aug_array = np.array(batch_rotate)

        return images_aug_array


    def normalise_images(self, imgs):        
        return (imgs - self.fold_mean)/self.fold_std


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.img_paths)


    def __getitem__(self, index):
            'Generates one sample of data'
            img_path = self.img_paths[index]
            ID = os.path.basename(img_path)
            img = tifffile.imread(img_path)
            img = np.moveaxis(img, 0, -1)

            assert(img.shape[0] == self.in_height)
            assert(img.shape[1] == self.in_width)

            img = self.normalise_images(img)

            if self.isTraining:
                img = np.squeeze(self.augment_data(img))
            
            img = np.moveaxis(img, -1, 0)

            # Get labels
            ID_reformat = ID.split('_')[0]
            ID_reformat = ID_reformat[:2] + '-' + ID_reformat[2:]
            labels = self.labels_df.loc[self.labels_df['METABRIC.ID'] == ID_reformat].values.tolist()

            labels_time = np.zeros(1)
            labels_censored = np.zeros(1)

            labels_time[0] = labels[0][-1]
            labels_censored[0] = int(labels[0][-2])

            clinical = self.clinical_df.loc[self.labels_df['METABRIC.ID'] == ID_reformat].values.tolist()
            
            # Scalar, 5-dim
            clinical = np.array(clinical[0])
            # One-hot, 4-dim        
            clinical_cat = self.clinical_df_cat.loc[self.labels_df['METABRIC.ID'] == ID_reformat].values.tolist()

            ct_1h = np.zeros(self.n_classes_cat[0])
            ct_1h[clinical_cat[0][0]] = 1
            rt_1h = np.zeros(self.n_classes_cat[1])
            rt_1h[clinical_cat[0][1]] = 1
            ht_1h = np.zeros(self.n_classes_cat[2])
            ht_1h[clinical_cat[0][2]] = 1
            er_1h = np.zeros(self.n_classes_cat[3])
            er_1h[clinical_cat[0][3]] = 1
            
            clinical = np.concatenate((clinical, ct_1h, rt_1h, ht_1h, er_1h))
            
            clinical_cat = np.array(clinical_cat[0])

            # Convert to tensor
            img_torch = torch.from_numpy(img).float()
            clinical_torch = torch.from_numpy(clinical).float()
            clinical_cat_torch = torch.from_numpy(clinical_cat).long()
            labels_torch = torch.from_numpy(labels_time).float()
            censored_torch = torch.from_numpy(labels_censored).long()

            if self.isTraining:
                return img_torch, clinical_torch, clinical_cat_torch, labels_torch, censored_torch
            else:
                return img_torch, clinical_torch, clinical_cat_torch, labels_torch, censored_torch, ID