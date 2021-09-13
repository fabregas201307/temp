#!/bin/bash

export AWS_PROFILE="crayon-site"    # change to your profile name

aws s3 cp s3://st-crayon-dev/tf-outputs/resnet50-acrlg-224x224-896x896-random-6-2020-09-02-19-28-08-000/output/model.tar.gz ./model_p1.tar.gz  # change the S3 path to your P1 model
tar -zxf model_p1.tar.gz
mv ./model ./model_p1
tar -cvzf model_and_code_p1.tar.gz code --directory=model_p1 1
aws s3 cp ./model_and_code_p1.tar.gz s3://st-crayon-dev/cli/inference/model/p1_model_and_inference_code.tar.gz  # don't change this s3 path
rm -rf model_p1
rm ./model_p1.tar.gz
rm ./model_and_code_p1.tar.gz

aws s3 cp s3://st-crayon-dev/tf-outputs/resnet50-ldal-224x224-500x500-tile-1x1--2020-09-10-21-03-17-871/output/model.tar.gz ./model_p2.tar.gz  # change the S3 path to your P2 model
tar -zxf model_p2.tar.gz
mv ./model ./model_p2
tar -cvzf model_and_code_p2.tar.gz code --directory=model_p2 1
aws s3 cp ./model_and_code_p2.tar.gz s3://st-crayon-dev/cli/inference/model/p2_model_and_inference_code.tar.gz  # don't change this s3 path
rm -rf model_p2
rm ./model_p2.tar.gz
rm ./model_and_code_p2.tar.gz
