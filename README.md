# White and grey matter segmentation on T1-weighted exvivo mouse spinal cord

[![DOI](https://zenodo.org/badge/587907110.svg)](https://doi.org/10.5281/zenodo.7772350)

https://user-images.githubusercontent.com/2482071/227744144-ff9b21c3-d757-4e4c-a990-f6d7bf3084b0.mov

## Citation

Publication linked to the dataset: Coming soon!

Publication linked to this model: see [CITATION.cff](./CITATION.cff)


## Project description

In this project, we trained a 3D nnU-Net for spinal cord white and grey matter segmentation. The data contains 22 mice with different number of chunks, for a total of 72 MRI 3D images. Each MRI image is T2-weighted, has a size of 200x200x500, with the following resolution: 0.05x0.05x0.05 mm. 

In order to train a 3D nnU-Net, the following steps were completed: 
- First, a total of 161 slices were labelled on various subjects. See [Notes](#notes) for details on the manual labeling.
- The slices were then extracted using the [extract_slices.py](./utils/extract_slices.py) function: it extracted both the slice from the MRI image as well as the mask's slice. These were gathered into a temporary dataset, on which a 2D nnU-Net model was trained to segment spinal cord white and grey matter. The inference was then performed using this model on the full 3D volume from the original dataset. 
- Then, a 3D nnU-Net was trained on the images, using the results from the previous inference as ground truth as well as using extracted slices (of shape (200x200x1)) and their manual segmentation. The inference, was again performed on the full zurich-mouse dataset. Going from a 2D nnU-Net to a 3D nnU-Net helped improved the continuity of the segmentation on the z-axis. 
- After that, we selected the best segmentation masks on the dataset totalling 31 images. For each of these images we noted that the top and bottom slices were often poorly annotated. Using the [crop_image_and_mask.py](./utils/crop_image_and_mask.py) script we removed these slices. The objective was to keep only qualitative annotations. 
- Finally, a 3D nnU-Net was trained on these qualitative image segmentations (31 images) with various dimension as well as annotated slices (161 images). The nnU-Net was trained on 1000 epochs, with "3d_fullres" configuration and on 5 folds. The best Dice score were the following (fold 0 : 0.9135, fold 1: 0.9083, fold 2: 0.9109 , fold 3: 0.9132, fold 4: 0.9173). 

For the packaging we decided to keep only fold 4 as it has the best dice score and all performed similarly in terms of final results as well as training evolution (meaning that the dataset is rather homogeneous). The reason for this is to avoid having to upload the full results model which weight around 5 GB and limit ourself to 250 MB. Also, inference is much longer when performed on 5 folds instead of 1 and results are comparable. 

This `README` file shows how to use the model which we trained to infer predictions. For information on how to retrain the same model, refer to this file [README_full_process.md](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/main/utils/README_full_process.md). 

## Installation

Clone the repository:
~~~
git clone https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1.git
cd model_seg_mouse-sc_wm-gm_t1
~~~

We recommend to use a virtual environment with python 3.9 to use nnUNet: 
~~~
conda create -n venv_nnunet python=3.9
~~~

We activate the environment:
~~~
conda activate venv_nnunet
~~~

Then install the required libraries:
~~~
pip install -r requirements.txt
~~~

## Perform predictions

To run an inference and obtain a segmentation, we advise using the following method (refer to `utils/README_full_process.md` for alternatives). 

Download the model `Dataset500_zurich_mouse.zip` from the release and unzip it. 
Use the `test.py` function:

To run on an entire dataset:
~~~
python test.py --path-dataset /path/to/test-dataset --path-out /path/to/output --path-model /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres
~~~

To run on individual(s) NIfTI image(s):
~~~
python test.py --path-images /path/to/image1 /path/to/image2 --path-out /path/to/output --path-model /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres
~~~

> The `nnUNetTrainer__nnUNetPlans__3d_fullres` folder is inside the `Dataset500_zurich_mouse` folder.

> To use GPU, add the flag `--use-gpu` in the previous command.

> To use mirroring (test-time) augmentation, add flag `--use-mirroring`. NOTE: Inference takes a long time when this is enabled. Default: False.

## Apply post-processing

nnU-Net v2 comes with the possiblity of performing post_processing on the segmentation images. This was not included in the run inference script as it doesn't bring notable change to the result. To run post-processing run the following script.

~~~
CUDA_VISIBLE_DEVICES=XX nnUNetv2_apply_postprocessing -i /seg/folder -o /output/folder -pp_pkl_file /path/to/postprocessing.pkl -np 8 -plans_json /path/to/post-processing/plans.json
~~~

> The file `postprocessing.pkl` is stored in `Dataset500_zurich_mouse/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl`.

> The file `plans.json` is stored in `Dataset500_zurich_mouse/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json`. 

## Notes

Procedure for ground truth mask creation: https://youtu.be/KVL-JzcSRTo