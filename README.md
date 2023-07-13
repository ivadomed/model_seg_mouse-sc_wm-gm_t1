# White and grey matter segmentation on T1-weighted exvivo mouse spinal cord

TO DO : record new video 
https://user-images.githubusercontent.com/2482071/227744144-ff9b21c3-d757-4e4c-a990-f6d7bf3084b0.mov

## Citation

TO DO : UPDATE THE CITATION

Publication linked to the dataset: Coming soon!

Publication linked to this model: see [CITATION.cff](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/CITATION.cff)


## Installation

Clone the repository:
~~~
git clone https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1.git
cd model_seg_mouse-sc_wm-gm_t1
~~~

We recommend installing a virtual environnment with python 3.9 installed (required for nnU-Net v2). Follow the instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) 

Then install the required libraries:
~~~
pip install -r requirements.txt
~~~

## Project description

In this project, we trained a 3D nnU-Net for spinal cord white and grey matter segmentation. The data used comes from the zurich-mouse dataset (INSERT LINK TO DATASET REF): it contains 22 subjects with different numbers of chunk for each totalling 72 MRI images. Each MRI images is T2 weighted, has the following dimension (200x200x500) and the following voxel size (0.05x0.05x0.05). 

In order to train a 3D nnU-Net, the following steps were completed: 
- First, a total of 161 slices were labelled on various subjects. (ADD MANUAL LABELLING PROCEDURE)
- The slices were then extracted using the [extract_slices.py](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/extract_slices.py) function: it extracted both the slice from the MRI image as well as the mask's slice. These were gathered into a temporary dataset, on which a 2D nnU-Net model was trained to segment spinal cord white and grey matter. The inference was then performed using this model on the full 3D volume from the original dataset. 
- Then, a 3D nnU-Net was trained on the images from the zurich-mouse dataset (of shape (200x200x500)), using the results from the previous inference as ground truth as well as using extracted slices (of shape (200x200x1)) and their manual segmentation. The inference, was again performed on the full zurich-mouse dataset. Going from a 2D nnU-Net to a 3D nnU-Net helped improved the continuity of the segmentation of the z-axis. 
- After that, we selected the best segmentation masks on the dataset totalling 31 images. For each of these images we noted that the top and bottom slices were often poorly annotated. Using the `crop_images.py` script we removed these slices. The objective was to keep only qualitative annotations. 
- Finally, a 3D nnU-Net was trained on these qualitative image segmentations (31 images) with various dimension as well as annotated slices (161 images). The nnU-Net was trained on 1000 epochs, with "3d_fullres" configuration and on 5 folds. The best Dice score were the following (fold 0 : 0.9135, fold 1: 0.9083, fold 2: 0.9109 , fold 3: 0.9132, fold 4: 0.9173). 

For the packaging we decided to keep only fold 4 as it has the best dice score and all performed simimarly in terms of final results as well as training evolution (meaning that the dataset is rather homogeneous). The reason for this is to avoid having to upload the full results model which weight around 5GB and limit ourself to 250 MB. Also, inference is much longer when performed on 5 folds instead of 1 and results are comparable. 

## Data pre-processing

Download dataset (internal git-annex): `zurich-mouse`

Then create the following folders in the right repo:

~~~
mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_results
~~~

**To extract slices**

In order to extract slices from the original file (dimension (500,500,200)) and their label, and to create a new file (dimesion (500,500,1)) in another dataset folder. To do so we use [extract_slices.py](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/extract_slices.py) in the following way:

~~~
python extract_slices.py --path-data /path/to/data --path-out /path/to/project/folder
~~~

**To crop images**

To crop both images and their respective mask in order to remove certain slices, use the following script [crop_image_and_mask.py](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/crop_image_and_mask.py): 

~~~
python crop_image_and_mask.py --folder-path /path/to/folder/  --file-name file_name.nii.gz --first-slice XXX --last-slice XXX
~~~

**To convert from BIDS to nnU-Net format**

Before using the nnU-Net model, we convert the dataset from the BIDS format to the nnU-Net fornat using [convert_bids_to_nnunet.py](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/convert_bids_to_nnunet.py). In this script, all the labeled data is used for training and the unlabeled-data is used for inference. 

~~~
python /path/to/repo/convert_bids_to_nnunet.py --path-data /path/to/data_extracted --path-out /path/to/nnUNet_raw --taskname TASK-NAME --tasknumber DATASET-ID
~~~

This will output a dataset called `DatasetDATASET-ID_TASK-NAME` in the `/nnUNet_raw` folder. (DATASET-ID has to be between 100 and 999).

## Data preprocessing

Before training the model, nnU-Net performs data preprocessing and checks the integrity of the dataset. 

To do so execute:
~~~
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

nnUNetv2_plan_and_preprocess -d DATASET-ID --verify_dataset_integrity
~~~

You will get the configuration plan for all four configurations (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres).
> In the case of the zurich_mouse dataset, nifti files are not fully annotated, therefore we use a 2d configuration.

## Train model

To train the model, use the following command:
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASET-ID CONFIG FOLD --npz
~~~

> Example for Dataset 101, on 2d config on fold 0: CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 101 2d 0 --npz

You can track the progress of the model with: 
~~~
nnUNet_results/DatasetDATASET-ID_TASK-NAME/nnUNetTrainer__nnUNetPlans__CONFIG/fold_FOLD/progress.png
~~~

## Run inference

To run an inference and obtain a segmentation, there are two ways to do so: 
- one option is to use a terminal command line
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_predict -i ./path/to/nnUNet_raw/DatasetDATASET-ID_TASK-NAME/imagesTs -o ./path/to/predictions -d DATASET_ID -c CONFIG --save_probabilities -chk checkpoint_best.pth -f FOLD
~~~

You can now access the predictions in the folder `./path/to/predictions`. 

- another option is to use the [test.py](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/test.py) python script. To do so, run the following line:

    Method 1 (when running on whole dataset):
~~~
python test.py --path-dataset /path/to/test-dataset --path-out /path/to/output --path-model /path/to/model
~~~
    Method 2 (when running on individual images):
~~~
python test.py --path-images /path/to/image1 /path/to/image2 --path-out /path/to/output --path-model /path/to/model
~~~

## Apply post-processing

nnU-Net v2 comes with the possiblity of performing post_processing on the segmentation images. This was not included in the run inference script as it doesn't bring notable change to the result. To run post-processing run the following script.

~~~
CUDA_VISIBLE_DEVICES=XX nnUNetv2_apply_postprocessing -i /seg/folder -o /output/folder -pp_pkl_file /path/to/postprocessing.pkl -np 8 -plans_json /path/to/post-processing/plans.json
~~~

## Notes

Procedure for ground truth mask creation: https://youtu.be/KVL-JzcSRTo

Before applying the model, make sure the image orientation is correct. More details [here](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/issues/25). 
