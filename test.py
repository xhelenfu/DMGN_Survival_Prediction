import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

from dataio.dataset_lidc import DatasetLIDC
from model.model import Network
from utils.utils import *

import numpy as np

def main(config):

    config_fold = config.config_file + str(config.fold_id) + '.json'
    json_opts = json_file_to_pyobj(config_fold)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    model_dir = experiment_path + '/' + json_opts.experiment_dirs.model_dir
    test_output_dir = experiment_path + '/' + json_opts.experiment_dirs.test_output_dir
    make_dir(test_output_dir)

    fold_mean = json_opts.data_params.fold_means
    fold_std = json_opts.data_params.fold_stds
    assert(len(fold_mean) == len(fold_std))
    n_markers = len(fold_mean)

    # Set up the model
    logging.info("Initialising model")
    model_opts = json_opts.model_params
    n_out_features = 1

    # Input variables and classes
    cont_cols = ['Age.At.Diagnosis', 'Ki67', 'EGFR', 'PR', 'HER2']
    cat_cols = ['CT', 'RT', 'HT', 'ER.Status']

    di_ct = {"NO/NA": 0, "ECMF": 1, "OTHER": 1, "AC": 1, "CAPE": 1, 
             "AC/CMF": 1, "CMF": 1, "PACL": 1, "FAC": 1}    
    di_rt = {"CW": 1, "NO/NA": 0, "CW-NODAL": 1, "NONE RECORDED IN LANTIS": 0}    
    di_ht = {"TAM": 1, "NO/NA": 0, "TAM/AI": 1, "AI": 1, "GNRHA": 1, 
             "OO": 1, "OTHER": 1, "Y": 1}   
    di_er = {"neg": 0, "pos": 1}    
    n_classes_cat = [2, 2, 2, 2] 

    model = Network(model_opts, n_out_features, n_markers,
                    1, device,
                    len(cont_cols), n_classes_cat)
    model = model.to(device) 

    # Dataloader
    logging.info("Preparing data")
    num_workers = json_opts.data_params.num_workers
    test_dataset = DatasetLIDC(json_opts.data_source, config.fold_id, fold_mean, fold_std,
                                json_opts.data_params.in_h, json_opts.data_params.in_w,
                                di_ct, di_rt, di_ht, di_er, cont_cols, cat_cols, n_classes_cat,
                                isTraining=False)
    test_loader = DataLoader(dataset=test_dataset, 
                              batch_size=1, 
                              shuffle=False, num_workers=num_workers)

    n_test_examples = len(test_loader)
    logging.info("Total number of testing examples: %d" %n_test_examples)

    # Get list of model files
    if config.test_epoch < 0:
        saved_model_paths, _ = get_files_list(model_dir, ['.pth'])
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_epochs = [(os.path.basename(x)).split('.')[0] for x in saved_model_paths]
        saved_model_epochs = [x.split('_')[-1] for x in saved_model_epochs]
        if config.test_epoch == -2:
            saved_model_epochs = np.array(saved_model_epochs, dtype='int')
        elif config.test_epoch == -1:
            saved_model_epochs = np.array(saved_model_epochs[-1], dtype='int')
            saved_model_epochs = [saved_model_epochs]
    else:
        saved_model_epochs = [config.test_epoch]

    logging.info("Begin testing")

    c_idx_epochs_avg = np.zeros(len(saved_model_epochs))

    for epoch_idx, test_epoch in enumerate(saved_model_epochs):

        y_true = np.zeros(n_test_examples)
        y_out = np.zeros(n_test_examples)
        death_true = np.zeros(n_test_examples)

        # Restore model
        load_path = model_dir + "/epoch_%d.pth" %(test_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        assert(epoch == test_epoch)
        print("Testing " + load_path)

        model = model.eval()

        # Write predictions to text file
        txt_path = test_output_dir + '/' + 'epoch_' + str(test_epoch) + '.txt'
        
        with open(txt_path, 'w') as output_file:

            for batch_idx, (batch_x, batch_c, batch_cat, batch_y, death_indicator, ID) in enumerate(test_loader):

                # Transfer to GPU
                batch_x, batch_c, batch_cat, batch_y = batch_x.to(device), batch_c.to(device), batch_cat.to(device), batch_y.to(device)
                death_indicator = death_indicator.to(device)

                # Forward pass
                final_pred, _ = model(batch_x, batch_c)

                # Labels, predictions per example
                y_true[batch_idx] = batch_y.squeeze().detach().cpu().numpy()
                y_out[batch_idx] = final_pred.squeeze().detach().cpu().numpy()
                death_true[batch_idx] = death_indicator.squeeze().detach().cpu().numpy()

                output_file.write(str(ID) + ' predict: ' + str(y_out[batch_idx]) + '\n')

            # Compute performance
            c_idx_epochs_avg[epoch_idx] = calc_concordance_index(y_out, death_true, y_true)

            print('C-index: mean ', np.around(c_idx_epochs_avg[epoch_idx],5))

            output_file.write('Overall Performance C-index\n')
            output_file.write(str(np.around(c_idx_epochs_avg[epoch_idx],5)))

    best_epoch_c_idx = np.argmax(c_idx_epochs_avg)

    print('Best C-index epoch %d mean %.5f' %(saved_model_epochs[best_epoch_c_idx], c_idx_epochs_avg[best_epoch_c_idx]))

    logging.info("Testing finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config', type=str,
                        help='config file path')
    parser.add_argument('--test_epoch', default=-2, type=int,
                        help='test model from this epoch, -1 for last, -2 for all')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')

    config = parser.parse_args()
    main(config)
