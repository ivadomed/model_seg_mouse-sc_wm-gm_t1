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

from utils.image import Image, change_orientation

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


def splitext(fname):
        """
        Split a fname (folder/file + ext) into a folder/file and extension.

        Note: for .nii.gz the extension is understandably .nii.gz, not .gz
        (``os.path.splitext()`` would want to do the latter, hence the special case).
        Taken (shamelessly) from: https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
        """
        dir, filename = os.path.split(fname)
        for special_ext in ['.nii.gz', '.tar.gz']:
            if filename.endswith(special_ext):
                stem, ext = filename[:-len(special_ext)], special_ext
                return os.path.join(dir, stem), ext
        # If no special case, behaves like the regular splitext
        stem, ext = os.path.splitext(filename)
        return os.path.join(dir, stem), ext


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension. Taken (shamelessly) from:
    https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py and adapted

    :param fname: absolute or relative file name. Example: t2.nii.gz
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:

    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def main():

    parser = get_parser()
    args = parser.parse_args()

    # Create the output path
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)
    
    fname_file = args.path_image

    # Create temporary directory in the temp to store the reoriented images
    prefix = f"sciseg_prediction_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix)

    # Copy the file to the temporary directory using shutil.copyfile
    fname_file_tmp = os.path.join(tmpdir, os.path.basename(fname_file))
    shutil.copyfile(fname_file, fname_file_tmp)
    print(f'Copied {fname_file} to {fname_file_tmp}')

    # Change orientation to LPI
    image_temp = Image(fname_file_tmp)
    # Store original orientation
    orig_orientation = image_temp.orientation

    # Reorient the image to RPI orientation if not already in RPI
    if orig_orientation != 'LPI':
        # reorient the image to RPI using SCT
        os.system('sct_image -i {} -setorient LPI -o {}'.format(fname_file_tmp, fname_file_tmp))
    # image_temp.change_orientation('LPI')
    # image_temp.save(fname_file_tmp)

    # NOTE: for individual images, the _0000 suffix is not needed.
    # BUT, the images should be in a list of lists
    fname_file_tmp_list = [[fname_file_tmp]]

    # Use all the folds available in the model folder by default
    folds_avail = [int(f.split('_')[-1]) for f in os.listdir(args.path_model) if f.startswith('fold_')]

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(fname_file_tmp, '_pred')))
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

    print('Deleting the temporary folder...')
    # Delete the temporary folder
    shutil.rmtree(tmpdir)

    print('----------------------------------------------------')
    print('Results can be found in: {}'.format(args.path_out))
    print('----------------------------------------------------')

if __name__ == '__main__':
    main()
