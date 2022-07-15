import os
import re


def sorted_alphanumeric(data):
    """
    Alphanumerically sort a list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def get_files_list(path, ext_array=['.tif']):
    """
    Get all files in a directory with a specific extension
    """
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list


def get_data_dirs_split(patient_subset_txt, img_data_dir):
    """
    Get list of samples in current fold split
    """
    with open(patient_subset_txt, 'r') as f:
        patient_subset = f.read().splitlines()
        
    img_list = [(img_data_dir + '/' + x) for x in patient_subset]
    img_list = sorted_alphanumeric(img_list)
    
    return img_list

