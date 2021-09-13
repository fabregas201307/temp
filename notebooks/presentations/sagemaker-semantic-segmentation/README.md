# Training, Deploying, and Testing the Segmentation Model

The notebooks are split into three sections: one for training, one for deploying an endpoint, and the last for testing an endpoint to get segmentation mask predictions for images.

## Initial setup

Please have python 3.6+ installed and run within this root directory:
```
pip install -r requirements.txt
```

## Notebook descriptions

  1. `semantic_seg_model_training.ipynb` : This notebook provide steps for setting up an experiment, tracking it, and training the aws semantic segmentation model. A single experiment is for a specific set of data and task (eg. performing semantic segmentation on labelbox annotated images). This is the high-level task and each trial are the sub-tasks which are training runs done with a specific set of hyperparameters. Naming convention we used for trials are `ss-labelbox-I<Image Size>-B<Base Size>-C<Crop size>-E<Epochs>-<Decoder>`. An example is: `ss-labelbox-I512-B512-C224-E20-Deeplab`.
  2. `semantic_seg_model_deployment.ipynb` : This notebook provide steps for hosting a trained model using the python sdk. Models can be also deployed using sagemaker studio.
  3. `semantic_seg_individual_predictions.ipynb` This notebook provide steps for getting inferences from a deployed model.
  4. `semantic_seg_folder_predictions.ipynb` This notebook provide steps for getting predictions for an entire s3 bucket from a deployed model.
  5. `semantic_seg_easy_deploy_inference_local_images.ipynb` This notebook provide steps to deploy a notebook from a pre-defined configuration and get predictions from local images and masks.
  6. `semantic_seg_create_manifest.ipynb` This notebook creates the manifest files to enable training in `semantic_seg_model_training.ipynb`, but this only needs to be run once to define the manifest file. 
