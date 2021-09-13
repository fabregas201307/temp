This is an example of using Sagemaker to deploy a customized PyTorch instance segmentation model that is trained locally.
The example is based on this [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) that fine-tunes a pretrained mask RCNN model over a dataset to detect people in images.

Your SITE AWS IAM profile should be saved as `crayon-site` or default profile.

The notebook implement the following process:
1. Prepare training model and upload to a S3 bucket `crayon-sagemaker-test` in folder `pytorch-pennfudanped-deploy-only`.
2. Deploy the trained model to Sagemaker with inference script in `trained_model/code/inference.py`, and create a simple UI in notebook to call the deployed endpoint.