"""Extract annotated slices from MRI image and segmentation mask

This python script extracts the segmentations masks slices which are annotated as well as the corresponding mri image slice.
It creates a new folder with the BIDS dataset format.

Example of run:

    $ python extract_slices.py --path-data /path/to/data --path-out /path/to/project/folder

Arguments:

    --path-data : Path to BIDS structured dataset.
    --path-out : Path to output directory.
    
Todo:
    * 

Pierre-Louis Benveniste
"""

import os
import numpy as np
from nibabel import load, Nifti1Image, save
from tqdm import tqdm
import argparse
import pathlib
from pathlib import Path
from time import time


# parse command line arguments
parser = argparse.ArgumentParser(description='Extract annotated slices from dataset.')
parser.add_argument('--path-data', required=True,
                    help='Path to BIDS structured dataset.')
parser.add_argument('--path-out', help='Path to output directory.', required=True)

args = parser.parse_args()

path_in_images = Path(args.path_data)
path_in_labels = Path(os.path.join(args.path_data, 'derivatives', 'manual_masks'))
path_out_images = Path(os.path.join(args.path_out, 'data_extracted'))
path_out_labels = Path(os.path.join(args.path_out, 'data_extracted','derivatives', 'manual_masks'))

if __name__ == '__main__':
    
    #create output folders
    pathlib.Path(path_out_images).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labels).mkdir(parents=True, exist_ok=True)


    folders_paths = [ f.path for f in os.scandir(path_in_images) if f.is_dir() and 'derivatives' not in f.path]
    
    #we iterate over folders
    total_count=0
    for folder in folders_paths:
        time0 = time()

        folder_name = folder.split('/')[-1]
        files_paths = [ f.path for f in os.scandir(Path(os.path.join(folder, 'anat'))) if '.nii.gz' in f.path]
        print("Processing folder: "+str(folder_name)+" ...")
        
        #we iterate over .nii.gz files
        for file in files_paths:
            #we can now access each .nii.gz file
            file_name = file.split('/')[-1]
            file_name_split = file_name.split('.')[0]
            #we open the image file
            mri = load(file)
            mri_full = np.asarray(mri.dataobj)
            #we create the mri folder for that specific mouse
            mri_folder_out = Path(os.path.join(path_out_images, folder_name, 'anat'))
            pathlib.Path(mri_folder_out).mkdir(parents=True, exist_ok=True)
            #we first save the entire nifti file
            img = Nifti1Image(mri_full, affine=mri.affine)
            mri_out_name = Path(os.path.join(mri_folder_out, file_name))
            save(img, str(mri_out_name))  
            
            #access of the label
            path_label_folder = Path(os.path.join(path_in_labels, folder_name, 'anat'))
            label_path_GM = [ f.path for f in os.scandir(path_label_folder) if '.nii.gz' in f.path and file_name_split in f.path and 'GM' in f.path]
            label_path_WM = [ f.path for f in os.scandir(path_label_folder) if '.nii.gz' in f.path and file_name_split in f.path and 'WM' in f.path]
            #we check if there is a label
            if len(label_path_GM)!=0:
                label_GM = load(label_path_GM[0])
                label_WM = load(label_path_WM[0])
                #Shape is supposed to be (200,200,500)
                nb_slices = np.asarray(label_GM.dataobj).shape[2]
                img_dimension= np.asarray(label_GM.dataobj).shape[:2]

                #Now we iterate over slices ##can be improved with np.where(sum(label_GM_slice) not 0)
                for slice_i in range(nb_slices):
                    label_GM_slice = np.asarray(label_GM.dataobj)[:,:,slice_i]
                    label_WM_slice = np.asarray(label_WM.dataobj)[:,:,slice_i]  
                    
                    #we check which slices are annotated (not blank) 
                    if np.sum(label_GM_slice)!=0 or np.sum(label_WM_slice)!=0 :
                        total_count+=1
                        #We create the label folder for that specific mouse
                        label_folder_out = Path(os.path.join(path_out_labels, folder_name, 'anat'))
                        pathlib.Path(label_folder_out).mkdir(parents=True, exist_ok=True)
 
                        #then we save the annotated MR slices
                        mri_extract = np.asarray(mri.dataobj)[:,:,slice_i]
                        img_extract = Nifti1Image(mri_extract, affine=mri.affine)
                        mouse,chunk = file_name_split.split('_',2)[0],file_name_split.split('_',2)[1]
                        out_name = '{}_{}-slice-{}_{}.{}'.format(mouse, chunk, slice_i,'T1w', 'nii.gz')
                        path_out_name = Path(os.path.join(mri_folder_out, out_name))
                        save(img_extract, str(path_out_name)) 
                        
                        #then we save the masks slices
                        GM_label_img = Nifti1Image(label_GM_slice, affine=label_GM.affine) 
                        WM_label_img = Nifti1Image(label_WM_slice, affine=label_WM.affine)
                        GM_out_name = '{}_{}-slice-{}_{}.{}'.format(mouse,chunk, slice_i, 'T1w_label-GM_mask','nii.gz')
                        WM_out_name = '{}_{}-slice-{}_{}.{}'.format(mouse,chunk, slice_i, 'T1w_label-WM_mask','nii.gz')
                        GM_path_out_name = Path(os.path.join(label_folder_out, GM_out_name))
                        WM_path_out_name = Path(os.path.join(label_folder_out, WM_out_name))
                        save(GM_label_img, str(GM_path_out_name)) 
                        save(WM_label_img, str(WM_path_out_name)) 
        print("Done ! It took " + str(time()-time0)+"sec")

    print('---  Finished: extracted '+ str(total_count)+' slices  ---')