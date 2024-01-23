"""Inference of 3D nnU-Net model

This python script runs inference of the 3D nnU-Net model on individual nifti images.  

Example of run:
        $ python test.py --path-images /path/to/image1 --path-out /path/to/output --path-model /path/to/model 

Arguments:
    --path-images : List of images to segment. Use this argument only if you want predict on a single image or list of invidiual images
    --path-out : Path to output directory
    --path-model : Path to the model directory. This folder should contain individual folders like fold_0, fold_1, etc.'
    --use-gpu : Use GPU for inference. Default: False
    --use-mirroring : Use mirroring (test-time) augmentation for prediction. NOTE: Inference takes a long time when this is enabled. Default: False
    
Todo:
    * 

Script inspired by script from Naga Karthik.
Pierre-Louis Benveniste
"""
import os
import shutil
import subprocess
import argparse
import datetime

import torch
import glob
import time
import tempfile
import numpy as np

from utils.image import Image, change_orientation
import nibabel as nib

# We define the environment variables here to avoid a warning from nnunetv2
os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results']="./nnUNet_results"

# Import for nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment images using nnUNet')
    parser.add_argument('--path-image', default=None,type=str)
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--path-model', required=True, 
                        help='Path to the model directory. This folder should contain individual folders '
                        'like fold_0, fold_1, etc.',)
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    parser.add_argument('--use-mirroring', action='store_true', default=False,
                        help='Use mirroring (test-time) augmentation for prediction. '
                        'NOTE: Inference takes a long time when this is enabled. Default: False')

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    # Create the output path
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)
    
    fname_file = args.path_image

    # Create temporary directory in the temp to store the reoriented images
    tmpdir = os.path.join(args.path_out, f"mouse_gm_wm_seg_pred_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_")
    print('Creating temporary directory')
    os.makedirs(tmpdir, exist_ok=True)

    # Copy the file to the temporary directory using shutil.copyfile
    fname_file_tmp = os.path.join(tmpdir, os.path.basename(fname_file))
    shutil.copyfile(fname_file, fname_file_tmp)
    print(f'Copied file to {fname_file_tmp}')

    # change resolution if needed
    img_updated_zooms = nib.load(fname_file_tmp)
    orig_resolution = list(img_updated_zooms.header.get_zooms())
    ratio = 0.05 / min(orig_resolution)
    if 1/ratio > 5 :
        dimensions = [orig_resolution[0] / 20, orig_resolution[1] /20, orig_resolution[2] /20]
        # Update the image header
        img_updated_zooms.header.set_zooms(dimensions)
        img_updated_zooms.set_sform(img_updated_zooms.get_qform())
        print("Resampling image to fit memory availability")
        nib.save(img_updated_zooms, fname_file_tmp)
        #nib.save(img_updated_zooms, os.path.join(args.path_out, "image_resized.nii.gz"))
    

    # Reorient the image to LPI orientation
    image_temp = Image(fname_file_tmp)
    orig_orientation = image_temp.orientation
    if orig_orientation != 'LPI':
        print('Reorienting image to LPI orientation')
        image_temp.change_orientation('LPI')
        image_temp.save(fname_file_tmp)
        #image_temp.save(os.path.join(args.path_out, "image_resized_rotated.nii.gz"))
        
    # NOTE: for individual images, the _0000 suffix is not needed.
    # BUT, the images should be in a list of lists
    fname_file_tmp_list = [[fname_file_tmp]]

    # Use all the folds available in the model folder by default
    folds_avail = [int(f.split('_')[-1]) for f in os.listdir(args.path_model) if f.startswith('fold_')]

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    os.mkdir(tmpdir_nnunet)

    # Run nnUNet prediction
    print('Starting inference...')
    start = time.time()

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.9,     # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,                      # applies gaussian noise and gaussian blur
        use_mirroring=False,                    # test time augmentation by mirroring on all axes
        perform_everything_on_gpu=True if args.use_gpu else False,
        device=torch.device('cuda') if args.use_gpu else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print('Running inference on device: {}'.format(predictor.device))

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(args.path_model),
        use_folds=folds_avail,
        checkpoint_name='checkpoint_best.pth',
    )
    print('Model loaded successfully. Fetching test data...')

    # NOTE: for individual files, the image should be in a list of lists
    predictor.predict_from_files(
        list_of_lists_or_source_folder=fname_file_tmp_list,
        output_folder_or_list_of_truncated_output_files=tmpdir_nnunet,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=8,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    
    end = time.time()
    print('Inference done.')
    total_time = end - start
    print('Total inference time: {} minute(s) {} seconds'.format(int(total_time // 60), int(round(total_time % 60))))

    # copy the prediction file to the output directory
    pred_file = glob.glob(os.path.join(tmpdir_nnunet, '*.nii.gz'))[0]
    out_file = os.path.join(args.path_out, str(args.path_image.split('/')[-1].split('.')[0]) + '_pred.nii.gz')
    shutil.copyfile(pred_file, out_file)
    #shutil.copyfile(pred_file, os.path.join(args.path_out, "model_ouput.nii.gz"))

    # change orientation back to original
    if orig_orientation != 'LPI':
        image_temp = Image(out_file)
        image_temp.change_orientation(orig_orientation)
        image_temp.save(out_file)
        #image_temp.save(os.path.join(args.path_out, "model_ouput_rotated.nii.gz"))

    # change resolution back to original
    if 1/ratio > 5 :
        img_reupdated_zooms = nib.load(out_file)
        img_reupdated_zooms.header.set_zooms(orig_resolution)
        img_reupdated_zooms.set_sform(img_reupdated_zooms.get_qform())
        nib.save(img_reupdated_zooms, out_file)
        #nib.save(img_reupdated_zooms, os.path.join(args.path_out, "model_ouput_rotated_resized.nii.gz"))

    print('Deleting the temporary folder...')
    # Delete the temporary folder
    shutil.rmtree(tmpdir)

    print('----------------------------------------------------')
    print('Results can be found in: {}'.format(args.path_out))
    print('----------------------------------------------------')

if __name__ == '__main__':
    main()
