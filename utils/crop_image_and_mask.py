"""Crops MRI image and respective mask

This python script crops MRI images and it's respective mask from a starting slice to an ending slice.

Example of run:

    $ python crop_image_and_mask.py --folder-path /path/to/folder/  --file-name file_name.nii.gz --first-slice XXX --last-slice XXX


Arguments:

    --folder-path : Path to the folder containing the image.
    --file-name : Name of the file to be cropped
    --first-slice : Beginning slice.
    --last-slice : Ending slice.

    
Todo:
    * 

Pierre-Louis Benveniste
"""


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import pathlib
import os

import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Crops images and their respective mask from a starting slice to an ending slice.')
parser.add_argument('--folder-path', required=True, type = str, help='Path to the folder containing the image.')
parser.add_argument('--file-name', required=True, type = str, help='Name of the file to be cropped')
parser.add_argument('--first-slice', required=True, type = int, help='Beginning slice')
parser.add_argument('--last-slice', required=True, type = int, help='Ending slice')

args = parser.parse_args()
folder_path = args.folder_path
file_name = args.file_name
first_slice = args.first_slice
last_slice = args.last_slice

if __name__ == '__main__':
    #Creation of the paths to the masks
    sub_mouse = file_name.split('_')[0]

    file_path = folder_path + '/' + sub_mouse + '/anat/' + file_name

    seg_file_WM_name = file_name.split('.')[0] + '_label-WM_mask.nii.gz'
    seg_file_GM_name = file_name.split('.')[0] + '_label-GM_mask.nii.gz'

    seg_GM_path = folder_path + '/derivatives/manual_masks/' + sub_mouse + '/anat/' + seg_file_GM_name
    seg_WM_path = folder_path + '/derivatives/manual_masks/' + sub_mouse + '/anat/' + seg_file_WM_name

    #Load image to be cropped
    img = nib.load(file_path)
    data = img.get_fdata()
    cropped_img = img.slicer[:,:,first_slice:last_slice]
    cropped_img.to_filename(file_path)
    print("done with ", file_name)

    #Load GM mask to be cropped
    img_GM = nib.load(seg_GM_path)
    data_GM = img_GM.get_fdata()
    cropped_img_GM = img_GM.slicer[:,:,first_slice:last_slice]
    cropped_img_GM.to_filename(seg_GM_path)
    print("done with ", seg_file_GM_name)

    #Load WM mask to be cropped
    img_WM = nib.load(seg_WM_path)
    data_WM = img_WM.get_fdata()
    cropped_img_WM = img_WM.slicer[:,:,first_slice:last_slice]
    cropped_img_WM.to_filename(seg_WM_path)
    print("done with ", seg_file_WM_name)
