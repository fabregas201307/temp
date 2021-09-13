# Training, Deploying, and Testing the U-Net Segmentation Model

## Initial setup

Please have python 3.6+ installed and run within this root directory:

```shell
pip install -r requirements.txt
```

If you want to run inference locally (see `unet_inference.ipynb`), TensorFlow (>=2.1) must be installed too.

```shell
pip install -r tensorflow
```

## Notebook descriptions

1. `unet_sagemaker_train_deploy.ipynb`: This notebook shows how to train and deploy a U-net model on Sagemaker.
2. `unet_sagemaker_experiment.ipynb`: This notebook shows how to train U-net model with multiple configurations of hyperparameters as experiment trial on Sagemaker.
3. `unet_inference.ipynb`: This notebooks shows how to use a trained U-net model for inference, either locally or through a deployed endpoint on Sagemaker.

## Difference from built-in Sagemaker segmentation model

This workflow use a customized TensorFlow model (U-net) built in a Sagemaker TensorFlow Docker container. It requires setting up the model (see `src/unet/models.py`), but provides more flexibility than the built-in model which is a blackbox. For example, we may customize how the training data is generated (see `src/unet/data.py`), therefore, this workflow does not require having separate preprocessing and postprocessing scripts. We may also customize details of the training process (see `src/unet/train.py`) and define our own metrics (see `src/unet/metrics.py`). The full accessibility to the model also enables us to get inference results of intermediate layers of the model which will be helpful for the active learning phase.
