This is an example of using Sagemaker to train and deploy a customized PyTorch instance segmentation model.
The example is based on this [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) that fine-tunes a pretrained mask RCNN model over a dataset to detect people in images.

Your SITE AWS IAM profile should be saved as `crayon-site` or default profile.

The notebook implement the following process:
1. Prepare training data and upload to a S3 bucket `crayon-sagemaker-test` in folder `pytorch-pennfudanped-train-deploy`.
2. Create and train a Sagemaker PyTorch model on AWS. The training script is defined by `sagemaker_src/pennfudanped.py`.
3. Deploy the trained model to AWS, and create a simple UI in notebook to call the deployed endpoint.