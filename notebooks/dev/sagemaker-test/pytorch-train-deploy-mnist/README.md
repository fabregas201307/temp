This is an example of using Sagemaker to train and deploy a customized PyTorch classification model.
The example is based on [this](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/pytorch_mnist), with minor modification on AWS configuration.

Your SITE AWS IAM profile should be saved as `crayon-site` or default profile.

The notebook implement the following process:
1. Prepare training data and upload to a S3 bucket `crayon-sagemaker-test` in folder `pytorch-mnist-train-deploy`.
2. Create and train a Sagemaker PyTorch model on AWS. The training script is defined by `mnist.py`.
3. Deploy the trained model to AWS, and create a simple UI in notebook to call the deployed endpoint.