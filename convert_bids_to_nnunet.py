# This script specifically converts a BIDS dataset split into train/val/test to the MSD dataset format for 
# training with nnUNet.
# The dataset split can be obtained in two ways: (1) from `split_dict.json` in ivadomed (see NOTE below), or, 
# (2) by creating a new split by running the `create_data_splits.py` file.
# This split will be used for creating the train and test folders for nnUNet. 
# 
# This is a NOT a do-it-all script. It is expected that the user will modify the script to specify the contrast
# suffixes and the label file names. The placeholders for user inputs are marked with "TODO" in the comments.
#  
# Usage: python convert_bids_to_nnunet.py --path-data /path/to/bids/dataset --path-out /path/to/output/directory
#                --taskname tSCIZurichLesions --tasknumber 502 --split-dict /path/to/ivadomed-split/dictionary
# 
# Format of the split dictionary:
# {
#     "train": [
#         "sub-zh63",
#         "sub-zh16",
#         "sub-zh81",
#         "sub-zh42",
#         "sub-zh49"
#     ],
#     "valid": [
#         "sub-zh79",
#         "sub-zh01"
#     ],
#     "test": [
#         "sub-zh02",
#         "sub-zh80"
#     ]
# }
#
# NOTE: The ivadomed dataset split can be found in "split_datasets.joblib" in the output folder of the ivadomed training.
# Use the following code snippet to load the joblib and save it as a json file to be used as input for this script:
# import joblib
# split_dict = joblib.load('/path/to/split_datasets.joblib')
# with open('/path/to/split_dict.json', 'w') as fp:
#     json.dump(split_dict, fp, indent=4) 
# 
# Authors: Julian McGinnis, Naga Karthik 


import argparse
import pathlib
from pathlib import Path
import json
import os
import shutil
from collections import OrderedDict

import nibabel as nib
import numpy as np


def binarize_label(subject_path, label_path):
    label_npy = nib.load(label_path).get_fdata()
    threshold = 1e-12
    label_npy = np.where(label_npy > threshold, 1, 0)
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)

def label_encoding(subject_path, label_path, label_gm_path, label_wm_path, threshold=1e-12):
    #Encoding of the label images with background=0 , GM=1 and WM=2
    #We consider that GM and WM don't overlap (maybe need to change that ?)

    #For grey matter
    label_gm_npy = nib.load(label_gm_path).get_fdata() 
    label_gm_npy = np.where(label_gm_npy > threshold, 1, 0)
    #For white matter
    label_wm_npy = nib.load(label_wm_path).get_fdata() 
    label_wm_npy = np.where(label_wm_npy > threshold, 2, 0)
    
    #Addition of both
    label_npy = label_gm_npy + label_wm_npy
    
    ref = nib.load(subject_path)
    label_bin = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    # overwrite the original label file with the binarized version
    nib.save(label_bin, label_path)


# parse command line arguments
parser = argparse.ArgumentParser(description='Convert BIDS-structured database to nnUNet format.')
parser.add_argument('--path-data', required=True,
                    help='Path to BIDS structured dataset. Accepts both cross-sectional and longitudinal datasets')
parser.add_argument('--path-out', help='Path to output directory.', required=True)
parser.add_argument('--taskname', default='MSSpineLesion', type=str,
                    help='Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus',)
parser.add_argument('--tasknumber', default=501,type=int, 
                    help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
parser.add_argument('--split-dict', help='Specify the splits using ivadomed dict, expecting a json file.', required=True)

args = parser.parse_args()

path_in_images = Path(args.path_data)
path_in_labels = Path(os.path.join(args.path_data, 'derivatives', 'manual_masks'))
path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Task{args.tasknumber}_{args.taskname}'))

# define paths for train and test folders 
path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

# we load both train and validation set into the train images as nnunet uses cross-fold-validation
train_images, train_labels = [], []
test_images, test_labels = [], []

# NOTE: if more than 1 contrast is to be used, then create train/test lists for each contrast. These additional
# contrasts are assumed to be co-registered so a single label file will be used for all contrasts. Hence, no need 
# to create separate lists for labels.
# Example: train_images_t2w, test_images_t2w = [], []

if __name__ == '__main__':

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    conversion_dict = {}
    dirs = sorted(list(path_in_images.glob('*/')))
    dirs = [str(x) for x in dirs]
    
    # filter out derivatives directory for raw images
    # ignore MAC .DS_Store files
    dirs = [k for k in dirs if 'sub' in k]
    dirs = [k for k in dirs if 'derivatives' not in k]
    dirs = [k for k in dirs if '.DS' not in k]


    scan_cnt_train, scan_cnt_test = 0, 0

    # load the dataset splits from ivadomed
    with open(args.split_dict) as f:
        splits = json.load(f)

    valid_train_imgs = []
    valid_test_imgs = []
    # NOTE: nnUNet does not require a separate validation folder, cross-validation is done internally 
    # from within the training set.
    valid_train_imgs.append(splits["train"])
    valid_train_imgs.append(splits["valid"])
    valid_test_imgs.append(splits["test"])

    # flatten the lists
    valid_train_imgs =[item for sublist in valid_train_imgs for item in sublist] 
    valid_test_imgs =[item for sublist in valid_test_imgs for item in sublist] 

    # assert number of training set images is equivalent to ivadomed
    for dir in dirs:  # iterate over subdirs

        # glob the session directories
        subdirs = sorted(list(Path(dir).glob('*')))
        subdirs = [k for k in subdirs if '.DS' not in str(k)]

        for subdir in subdirs:
            # TODO: Define the contrast to use here.
            # Examples: acq-ax_T2w.nii.gz, acq-sag_T2w.nii.gz, T2w.nii.gz, T1w.nii.gz
            # If you want to use only the sagittal image, use the following line:
            # image_file = sorted(list(subdir.rglob('*acq-sag_T2w.nii.gz')))[0]

            # find the corresponding image file
            image_files = sorted(list(subdir.rglob('*T1w.nii.gz')))
            
            #Identify common data path
            common = os.path.relpath(os.path.dirname(image_files[0]), args.path_data)
            # find the corresponding segmentation file
            label_path = os.path.join(path_in_labels, common)
            label_files = sorted(list(Path(label_path).rglob('*label*.nii.gz')))

            for image_file in image_files:
                subject_name = str(Path(image_file).name).split('_')[0]
                chunk_nb = str(Path(image_file).name).split('_')[1]
                
                label_file = [k for k in label_files if subject_name in str(k)]
                label_file = [k for k in label_files if chunk_nb in str(k)]

                
                
                if any(subject_name in word for word in valid_train_imgs) or any(subject_name in word for word in valid_test_imgs):
                    
                    #check if the label file is included for every chunk
                    if(len(label_file)!=2):
                        print("No segmentation mask for this subject's (" + str(subject_name) + ") chunk: " + str(chunk_nb))
                    else:
                        label_file_GM = [k for k in label_file if 'GM' in str(k)][0]
                        label_file_WM = [k for k in label_file if 'WM' in str(k)][0]
                        
                        if any(subject_name in word for word in valid_train_imgs):



                            scan_cnt_train+= 1
                            # create the new file names according to nnUNet's convention
                            # From nnUNet's documentation: 
                            # "Imaging files must therefore follow the following naming convention: case_identifier_XXXX.nii.gz. Hereby, XXXX is the modality identifier.
                            # Label files are saved as case_identifier.nii.gz". 
                            # More information can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md

                            image_file_nnunet = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0000.nii.gz')
                            label_file_nnunet_GM = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}_01.nii.gz')
                            label_file_nnunet_WM = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}_02.nii.gz')
                            label_file_nnunet = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}.nii.gz')
                            
                            train_images.append(str(image_file_nnunet))
                            train_labels.append(str(label_file_nnunet))
                            
                            # copy the files to new structure
                            shutil.copyfile(image_file, image_file_nnunet)
                            shutil.copyfile(label_file_GM, label_file_nnunet_GM)
                            shutil.copyfile(label_file_WM, label_file_nnunet_WM)
                            
                            #Encoding GM and WM label into one file
                            label_encoding(image_file_nnunet, label_file_nnunet, label_file_nnunet_GM, label_file_nnunet_WM)
                            os.remove(label_file_nnunet_GM)
                            os.remove(label_file_nnunet_WM)
                            
                            conversion_dict[str(os.path.abspath(image_file))] = image_file_nnunet
                            conversion_dict[str(os.path.abspath(label_file_GM))] = label_file_nnunet
                            conversion_dict[str(os.path.abspath(label_file_WM))] = label_file_nnunet
                        
                        else:
                            
                            # Repeat the above procedure for testing
                            scan_cnt_test += 1
                            # create the new convention names
                            image_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
                            label_file_nnunet_GM = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}_01.nii.gz')
                            label_file_nnunet_WM = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}_02.nii.gz')
                            label_file_nnunet = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')

                            test_images.append(str(image_file_nnunet))
                            test_labels.append(str(label_file_nnunet))

                            # copy the files to new structure
                            shutil.copyfile(image_file, image_file_nnunet)
                            shutil.copyfile(label_file_GM, label_file_nnunet_GM)
                            shutil.copyfile(label_file_WM, label_file_nnunet_WM)

                            #Encoding GM and WM label into one file
                            label_encoding(image_file_nnunet, label_file_nnunet, label_file_nnunet_GM, label_file_nnunet_WM)
                            os.remove(label_file_nnunet_GM)
                            os.remove(label_file_nnunet_WM)

                            conversion_dict[str(os.path.abspath(image_file))] = image_file_nnunet
                            conversion_dict[str(os.path.abspath(label_file_GM))] = label_file_nnunet
                            conversion_dict[str(os.path.abspath(label_file_WM))] = label_file_nnunet

                else:
                    print("Skipping file, could not be located in the specified split.", image_file)


    assert scan_cnt_train == len(valid_train_imgs), 'No. of train/val images does not correspond to ivadomed split dict.'
    assert scan_cnt_test == len(valid_test_imgs), 'No. of test images does not correspond to ivadomed split dict.'

    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)


    # c.f. dataset json generation. This contains the metadata for the dataset that nnUNet uses during preprocessing and training
    # general info : https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/utils.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.taskname
    json_dict['description'] = args.taskname
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"

    
    # Because only using one modality 
    json_dict['modality'] = {
            "0": "T1w",
        }
    
    # NOTE: 0 is always the background. Any class labels should start from 1.
    json_dict['labels'] = {
        "0": "background",
        "1": "GM",
        "2": "WM"
    }
   
    json_dict['numTraining'] = scan_cnt_train
    json_dict['numTest'] = scan_cnt_test

    json_dict['training'] = [{'image': str(train_labels[i]).replace("labelsTr", "imagesTr") , "label": train_labels[i] }
                                 for i in range(len(train_images))]
    # Note: See https://github.com/MIC-DKFZ/nnUNet/issues/407 for how this should be described
    json_dict['test'] = [str(test_labels[i]).replace("labelsTs", "imagesTs") for i in range(len(test_images))]

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)
