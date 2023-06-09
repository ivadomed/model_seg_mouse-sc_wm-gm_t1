# White and grey matter segmentation on T1-weighted exvivo mouse spinal cord

https://user-images.githubusercontent.com/2482071/227744144-ff9b21c3-d757-4e4c-a990-f6d7bf3084b0.mov

## Citation

Publication linked to the dataset: Coming soon!

Publication linked to this model: see [CITATION.cff](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/CITATION.cff)

## Installation

Clone the repository:
~~~
git clone https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1.git
cd model_seg_mouse-sc_wm-gm_t1
~~~

We recommend installing a virtual environnment with python 3.9 installed (required for nnU-Net v2). Follow the instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) 

~~~
pip install -r requirements.txt
~~~

## Data conversion

Download dataset (internal git-annex): `zurich-mouse`

Then create the following folders in the right repo:

~~~
mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_results
~~~
First, because only a few slices have been annotated, we extract them from the original file (dimension (500,500,200)) and their label, creating a new file (dimesion (500,500,1)). To do so we use [extract_slices.py](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/extract_slices.py).

~~~
python /path/to/repo/extract_slices.py --path-data /path/to/data --path-out /path/to/project/folder
~~~

Before using the nnU-Net model, we convert the dataset from the BIDS format to the nnU-Net fornat using [convert_bids_to_nnunet.py](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/plb/nnunet/convert_bids_to_nnunet.py). Because the dataset is relatively small, all the labeled data is used for training and the unlabeled-data is used for inference. 

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

> Example for Dataset 101, on 2d config with 5 folds: CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 101 2d 5 --npz

You can track the progress of the model with: 
~~~
nnUNet_results/DatasetDATASET-ID_TASK-NAME/nnUNetTrainer__nnUNetPlans__CONFIG/fold_FOLD/progress.png
~~~

SHOW AN EXAMPLE OF RUN

## Run inference

To run an inference and obtain prediction, do :
~~~
CUDA_VISIBLE_DEVICES=XXX nnUNetv2_predict -i ./path/to/nnUNet_raw/DatasetDATASET-ID_TASK-NAME/imagesTs -o ./path/to/predictions -d DATASET_ID -c CONFIG --save_probabilities -chk checkpoint_best.pth -f FOLD
~~~

You can now access the predictions in the folder `./path/to/predictions`. 

SHOW AN EXAMPLE OF PREDICTION

## Notes

Procedure for ground truth mask creation: https://youtu.be/KVL-JzcSRTo

Before applying the model, make sure the image orientation is correct. More details [here](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/issues/25). 
