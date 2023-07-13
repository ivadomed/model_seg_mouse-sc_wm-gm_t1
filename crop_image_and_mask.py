import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import pathlib
import os

import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Converts the 3D nnU-Net segmentations back to the format of the original zurich datasets and stores them in another folder insteafd of the /manual_masks.')
parser.add_argument('--folder-path', required=True, type = str, help='Folder path')
parser.add_argument('--file-name', required=True, type = str, help='Folder path')
parser.add_argument('--first-slice', required=True, type = int, help='Beginning slice')
parser.add_argument('--last-slice', required=True, type = int, help='Ending slice')

args = parser.parse_args()
folder_path = args.folder_path
file_name = args.file_name
first_slice = args.first_slice
last_slice = args.last_slice

sub_mouse = file_name.split('_')[0]

file_path = folder_path + '/' + sub_mouse + '/anat/' + file_name

seg_file_WM_name = file_name.split('.')[0] + '_label-WM_mask.nii.gz'
seg_file_GM_name = file_name.split('.')[0] + '_label-GM_mask.nii.gz'

seg_GM_path = folder_path + '/derivatives/manual_masks/' + sub_mouse + '/anat/' + seg_file_GM_name
seg_WM_path = folder_path + '/derivatives/manual_masks/' + sub_mouse + '/anat/' + seg_file_WM_name

img = nib.load(file_path)
data = img.get_fdata()
cropped_img = img.slicer[:,:,first_slice:last_slice]
cropped_img.to_filename(file_path)
print("done with ", file_name)

img_GM = nib.load(seg_GM_path)
data_GM = img_GM.get_fdata()
cropped_img_GM = img_GM.slicer[:,:,first_slice:last_slice]
cropped_img_GM.to_filename(seg_GM_path)
print("done with ", seg_file_GM_name)

img_WM = nib.load(seg_WM_path)
data_WM = img_WM.get_fdata()
cropped_img_WM = img_WM.slicer[:,:,first_slice:last_slice]
cropped_img_WM.to_filename(seg_WM_path)
print("done with ", seg_file_WM_name)
