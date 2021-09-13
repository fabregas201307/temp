# `stcrayon` - SITE CLI

`stcrayon` is a command line interface (CLI) of the SITE AI automation project.

## Installing

```bash
git clone git@github.com:SiteTechnologies/ai-automation.git
cd ai-automation/src/site_cli
pip install .
```

## Preparing AWS environment

### Create docker image for script processing

The preprocessing and postprocessing for model inference is implemented with [SageMaker script processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html). A docker image must be created on Amazon ECR so that the script processing may run in a responding container.

Run the following command to

1. Build the docker image locally;
2. Create an Amazon ECR repository;
3. Push the image to Amazon ECR.

```bash
cd prepare_aws
source ./build_ecr_image.sh
```

Note you must edit the setting of environment variable `AWS_PROFILE` according to your local AWS configuration if `crayon-site` is not your correct profile name.

### Adding I/O handlers to trained model for batch transform

Model inference cross preprocessed crops is implemented with [SageMaker batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-batch-transform.html). A TensorFlow model trained on SageMaker and saved with [SavedModel](https://www.tensorflow.org/guide/saved_model) format must be bundled with proper I/O handlers that are compatible with the preprocessing and postprocessing.

Run the following command to

1. Download a train TensorFlow model in SavedModel format from S3;
2. Bundle the model with code of I/O handlers;
3. Upload the model plus I/O handlers to S3.

```bash
cd prepare_aws
source ./add_io_handler_to_model.sh
```

Note you must edit the setting of environment variable `AWS_PROFILE` according to your local AWS configuration if `crayon-site` is not your correct profile name.

## Quickstart

```bash
stcrayon --help
```
