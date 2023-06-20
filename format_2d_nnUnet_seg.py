import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import pathlib
import os

import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Converts the 2D nnU-Net segmentations back to the format of the original zurich datasets and stores them in another folder insteaf of the /manual_masks.')
parser.add_argument('--path-conversion-dict', required=True, type = str, help='Path to the json conversion dicionnary to identify corresponding original image to each segmentation')
parser.add_argument('--path-segmentation-folder', help='Path to the segmentation folder.', type =str, required=True),
parser.add_argument('--path-dataset', help='Path to the dataset folder.', type =str, required=True),
parser.add_argument('--mask_name', default='2d_nnUNet_masks', type=str, help='Name of the folder created at same location as `manual_masks` folder')

#Args assigned to variables
args = parser.parse_args()
conv_dict_path = args.path_conversion_dict
seg_path = Path(args.path_segmentation_folder)
dataset_path = Path(args.path_dataset)
mask_folder_name = args.mask_name

if __name__ == '__main__':
    #Get the list of all full volume (not the slice extracted)
    f=open(conv_dict_path)
    conv_dict = json.load(f)
    list_3d_files = []
    for element in conv_dict:
        if 'slice' not in element:
            list_3d_files.append(element)

    #Get the list of segmentations
    seg_files = sorted(list(seg_path.rglob('*.nii.gz')))

    #Initiate the new mask folder in the datasets
    out_path =  Path(os.path.join(dataset_path, 'derivatives', mask_folder_name))
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    #iterate over every full volume
    for element in list_3d_files:
        #get the volume info (mouse nb and chunk nb)
        sub_mouse_chunk = element.split('/')[-1].split('.')[0]
        sub_mouse = sub_mouse_chunk.split('_')[0]
        #Create mouse sub folder
        sub_mouse_path = Path(os.path.join(out_path, sub_mouse, 'anat'))
        pathlib.Path(sub_mouse_path).mkdir(parents=True, exist_ok=True)
        chunk = sub_mouse_chunk.split('_')[1]
        #Get segmentation and load data
        seg_file_name = conv_dict[element].split('/')[-1].split('.')[0].rsplit('_',1)[0]
        seg_file_path = [k for k in seg_files if seg_file_name in str(k)][0]
        #Build and save GM mask
        seg_file = nib.load(seg_file_path)
        seg_data_GM = seg_file.get_fdata()
        seg_data_GM[seg_data_GM==2]=0.0
        final_img_GM = nib.Nifti1Image(seg_data_GM, affine=seg_file.affine) 
        file_path_out_GM = Path(os.path.join(sub_mouse_path, sub_mouse_chunk + '_label-GM_mask.nii.gz'))
        nib.save(final_img_GM, file_path_out_GM) 
        #Build and save WM mask
        seg_file = nib.load(seg_file_path)
        seg_data_WM = seg_file.get_fdata()
        seg_data_WM[seg_data_WM==1]=0.0
        seg_data_WM[seg_data_WM==2]=1
        final_img_WM = nib.Nifti1Image(seg_data_WM, affine=seg_file.affine) 
        file_path_out_WM = Path(os.path.join(sub_mouse_path, sub_mouse_chunk + '_label-WM_mask.nii.gz'))
        nib.save(final_img_WM, file_path_out_WM) 

        print("Done with ", sub_mouse_chunk)
    print("Finished")
