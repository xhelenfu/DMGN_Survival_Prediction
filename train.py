import argparse
import logging
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
    if config.resume_epoch == None:
        make_new = True 
    else:
        make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir, config.fold_id)
    experiment_path = 'experiments' + '/' + timestamp
    make_dir(experiment_path + '/' + json_opts.experiment_dirs.model_dir)

    fold_mean = json_opts.data_params.fold_means
    fold_std = json_opts.data_params.fold_stds
    assert(len(fold_mean) == len(fold_std))
    n_markers = len(fold_mean)

    # Set up the model
    logging.info("Initialising model")
    model_opts = json_opts.model_params
    n_out_features = 1
    n_classes = [1]*n_out_features

    # Input variables and class mappings for categorical variables
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
                    json_opts.training_params.batch_size, device,
                    len(cont_cols), n_classes_cat)
    model = model.to(device)

    # Dataloader
    logging.info("Preparing data")
    num_workers = json_opts.data_params.num_workers
    train_dataset = DatasetLIDC(json_opts.data_source, config.fold_id, fold_mean, fold_std,
                                json_opts.data_params.in_h, json_opts.data_params.in_w,
                                di_ct, di_rt, di_ht, di_er, cont_cols, cat_cols, n_classes_cat, 
                                isTraining=True)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=json_opts.training_params.batch_size, 
                              shuffle=True, num_workers=num_workers, drop_last=True)

    n_train_examples = len(train_loader)
    logging.info("Total number of training examples: %d" %n_train_examples)

    # Auxiliary losses and optimiser
    criterion_mae = torch.nn.MSELoss(reduction='sum')
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=json_opts.training_params.learning_rate, 
                                 betas=(json_opts.training_params.beta1, 
                                        json_opts.training_params.beta2),
                                 weight_decay=json_opts.training_params.l2_reg_alpha)

    if config.resume_epoch != None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if config.resume_epoch != None:
        load_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                    "/epoch_%d.pth" %(config.resume_epoch)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        assert(epoch == config.resume_epoch)
        print("Resume training, successfully loaded " + load_path)

    logging.info("Begin training")

    model = model.train()

    for epoch in range(initial_epoch, json_opts.training_params.total_epochs):
        epoch_train_loss = 0.

        for _, (batch_x, batch_c, batch_cat, batch_y, death_indicator) in enumerate(train_loader):

            # Transfer to GPU
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_c, batch_cat = batch_c.to(device), batch_cat.to(device)
            death_indicator = death_indicator.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            final_pred, aux_pred = model(batch_x, batch_c)

            # Optimisation
            if torch.sum(death_indicator) > 0.0:
                losses = []
                for lx in range(n_out_features):
                    end_idx = int(np.sum(n_classes[:lx+1]))
                    start_idx = int(end_idx - n_classes[lx])
                    losses.append(PartialLogLikelihood(final_pred[:,start_idx:end_idx], 
                                                       death_indicator, 'noties'))
                
                loss = sum(losses)[0]

                aux_loss = criterion_mae(aux_pred[:,:len(cont_cols)], 
                                         batch_c[:,:len(cont_cols)]).squeeze()
                loss += json_opts.training_params.lambda_aux*aux_loss

                for lx in range(len(n_classes_cat)):
                    end_idx = int(np.sum(n_classes_cat[:lx+1])) + len(cont_cols)
                    start_idx = int(end_idx - n_classes_cat[lx])
                    aux_loss = criterion_ce(aux_pred[:,start_idx:end_idx], batch_cat[:,lx]).squeeze()
                    loss += json_opts.training_params.lambda_aux*aux_loss

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.detach().cpu().numpy()
           
                
        # Save model
        if (epoch % json_opts.save_freqs.model_freq) == 0:
            save_path = experiment_path + '/' + json_opts.experiment_dirs.model_dir + \
                        "/epoch_%d.pth" %(epoch+1)
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_path)
            logging.info("Model saved: %s" % save_path)

        # Print training loss every epoch
        print('Epoch[{}/{}], total loss:{:.4f}'.format(epoch+1, json_opts.training_params.total_epochs, 
                                                       epoch_train_loss))

    logging.info("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', default='configs/config', type=str,
                        help='config file path')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='resume training from this epoch, set to None for new training')
    parser.add_argument('--fold_id', default=1, type=int,
                        help='which cross-validation fold')

    config = parser.parse_args()
    main(config)
