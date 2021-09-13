#!/bin/bash

export AWS_PROFILE="crayon-site"    # change to your profile name
export account_id="395166463292"
export region="us-east-2"
export ecr_repository="sitecli-inference-processing-container"
export tag="latest"
export processing_repository_uri="${account_id}.dkr.ecr.${region}.amazonaws.com/${ecr_repository}:${tag}"

docker build -t $ecr_repository docker

aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $account_id.dkr.ecr.$region.amazonaws.com
aws ecr create-repository --repository-name $ecr_repository

docker tag $ecr_repository:$tag $processing_repository_uri
docker push $processing_repository_uri
