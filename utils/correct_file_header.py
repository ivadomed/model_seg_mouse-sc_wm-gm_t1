""" Correct the file header of nifti files

This script corrects the file header of nifti files.
It was specifically written to correct the file header of the nifti files when the resolution of the nifti files is not correct. 

Example of run:
        $ python utils/correct_file_header.py --path-image /path/to/image --path-out /path/to/output --resolution 0.05 0.05 0.05

Arguments:
    --path-image : Path to the individual image to segment.
    --path-out : Path to output directory
    --resolution : Resolution of the nifti file. Default: 0.05 0.05 0.05

Returns:
    None

Todo:
    *

Pierre-Louis Benveniste
"""

import os
import argparse
import nibabel as nib


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment images using nnUNet')
    parser.add_argument('--path-image', default=None,type=str)
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--resolution', default=[0.05, 0.05, 0.05], type=float, nargs=3,
                        help='Resolution of the nifti file (separated by spaces). Default: 0.05 0.05 0.05')
    return parser

def main():
    """
    This is the main function of the script.
    
    Returns:
        None
    """

    # Get the arguments
    parser = get_parser()
    args = parser.parse_args()
    path_image = args.path_image
    path_out = args.path_out
    resolution = args.resolution

    # Create the output directory if it does not exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Create the file out path
    path_out = os.path.join(path_out, os.path.basename(path_image))

    # Load the image
    image = nib.load(path_image)
    image.header.set_zooms(resolution)
    image.set_sform(image.get_qform())
    nib.save(image, path_out)
    print('File header corrected.')

    # Print output advice
    print("To view the image in FSLeyes, run the following command:")
    print("fsleyes {}".format(path_out))

    return None


if __name__ == '__main__':
    main()