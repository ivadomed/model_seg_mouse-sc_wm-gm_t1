# White and grey matter segmentation on T1-weighted exvivo mouse spinal cord

https://user-images.githubusercontent.com/2482071/227744144-ff9b21c3-d757-4e4c-a990-f6d7bf3084b0.mov

## Citation

Publication linked to the dataset: https://pubmed.ncbi.nlm.nih.gov/35585865/

Publication linked to this model: see [CITATION.cff](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/main/CITATION.cff)

## Installation

Clone the repository:
~~~
git clone https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1.git
cd model_seg_mouse-sc_wm-gm_t1
~~~

We recommend installing a [virtual environment](https://docs.python.org/3/library/venv.html). Once installed, activate it and run:
~~~
pip install -r requirements.txt
~~~

## Train model

Dataset (internal git-annex): `zurich-mouse`

Procedure for ground truth mask creation: https://youtu.be/KVL-JzcSRTo

Example of training using GPU #0 and wandb group called "awesome-model" (for live monitoring):
~~~
export CUDA_VISIBLE_DEVICES="0"; export WANDB_RUN_GROUP="GROUP_NAME"; python train.py
~~~

## Test model

~~~
python test.py -i NIFTI_IMAGE
~~~

## Notes

Before applying the model, make sure the image orientation is correct. More details [here](https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/issues/25). 
