# Deep Multimodal Graph-Based Network for Survival Prediction from Highly Multiplexed Images and Patient Variables

Xiaohang Fu, Ellis Patrick, Jean Y.H. Yang, and Jinman Kim (2022).
doi: https://doi.org/10.1101/2022.07.19.500604

Implementation of the proposed deep learning model in Python and PyTorch. 

Personalised prediction of breast cancer survival using multimodal data (imaging mass cytometry images and patient variables).

*We introduce a deep multimodal graph-based network (DMGN) that integrates IMC images and multiple patient variables (PVs) for end-to-end survival prediction of breast cancer. We propose a graph-based module that aggregates deep image features from different spatial regions across the image with all PVs. We propose another module to automatically generate embeddings specialised for each PV to support multimodal aggregation. We show that our modules consistently enhance survival prediction performance using two public datasets, and that our DMGN outperforms state-of-the-art methods at survival prediction.*

![Alt text](Fig.png?raw=true)

## Requirements

Developed using Python ``3.7.8``, PyTorch ``1.5.0``, and PyTorch Geometric ``1.7.0``

Run ``pip install -r requirements.txt`` in your shell to install additional dependencies.

## Data

The METABRIC dataset (Ali et al. 2020; Curtis et al. 2012) may be obtained from https://idr.openmicroscopy.org/ (accession code idr0076).

#### IMC images:
- Resized to **176 x 176** lateral resolution using bilinear interpolation.
- .tiff files containing all the marker channels of each sample.

#### Labels and patient variables:
- .csv file containing the ground truth survival time, binary death event label, and 9 patient variables including 5 clinical features (age at diagnosis, ER status, chemotherapy indicator (CT), hormone treatment indicator (HT), and radiotherapy indicator (RT)), and 4 gene indicators (Ki67, EGFR, PR, and HER2).
- File contains 12 columns. First column is METABRIC.ID, followed by death event label, survival time (T), Age.At.Diagnosis, ER.Status, CT, HT, RT, Ki67, EGFR, PR, and HER2.
- Column names and data:
    - **METABRIC.ID**: MB-0000, MB-0002, etc.
    - **Death**: 0 or 1 
    - **T**: integer, survival days
    - **Age.At.Diagnosis**: scalar
    - **ER.Status**: pos or neg
    - **CT**: NO/NA, ECMF, OTHER, AC, CAPE, AC/CMF, CMF, PACL, or FAC 
    - **HT**: TAM, NO/NA, TAM/AI, AI, GNRHA, OO, OTHER, or Y   
    - **RT**: CW, NO/NA, CW-NODAL, or NONE RECORDED IN LANTIS    
    - **Ki67, EGFR, PR, and HER2**: scalar

#### Cross-validation fold splits:
- .txt files each containing a list of image filenames for different cross-validation folds.
- For example for fold 1: `1_train.txt`, and `1_test.txt`.
- METABRIC filenames are in the format: MB0188_1_537_fullstack.tiff, MB0453_1_468_fullstack.tiff, etc.

## Operation

Modify hyperparameters and locations of data files in the .json config file inside **``/configs``**.

Train the model using **``train.py``**.

Additional arguments for training:
- ``--config_file`` — path to config file
- ``--fold_id`` — cross-validation fold number
- ``--resume_epoch`` — ``None`` for train from scratch, or int number (e.g., 5) to resume training from saved model

Test the model using **``test.py``**.

Additional arguments for testing:
- ``--config_file`` — path to config file
- ``--fold_id`` — cross-validation fold number
- ``--test_epoch`` — which epoch to test (int number, -1 for latest saved model, or -2 for all saved models)
