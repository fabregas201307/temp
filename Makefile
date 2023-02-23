RELEASE_NAME=airflow-dashboard
IMAGE_NAME=myimage
SHA:=$(if $(SHA),$(SHA),$(shell git rev-parse --short HEAD))
GITREF:=$(if $(GITREF),$(GITREF),$(shell git rev-parse --abbrev-ref HEAD | sed 's/[^a-Z0-9-]/-/g' | sed 's/feature-//'))
FULL_IMAGE_TAG:=${IMAGE_NAME}:gc-${SHA}
BUILD_ENV:=$(if $(BUILD_ENV),$(BUILD_ENV),dev)
REVIEW_APP_NAME=sde-${GITREF}

## build and tag docker image
docker/build:
    $(info FULL_IMAGE_TAG=${FULL_IMAGE_TAG}...)
    $(info $(shell git log --online | tac | tail -1))
    docker build -t '${FULL_IMAGE_TAG}' .
## push docker image
docker/push:
    docker push "${FULL_IMAGE_TAG}"

check-env:
ifeq ($(BUILD_ENV), prod)
KUBE_CONTEXT=bam-dataprod-ue1-p-eks
NAMESPACE=sector-data-engineering-dev
endif
ifeq ($(BUILD_ENV), dev)
KUBE_CONTEXT=bam-dataprod-ue1-d-eks
NAMESPACE=sector-data-engineering-dev
endif

## helm install
helm/install: check-env
    helm install library/bam-simple-webapp --name $(RELEASE_NAME) \
     --namespace $(NAMESPACE) \
     --wait \
     --kube-context=$(KUBE_CONTEXT) \
     --namespace $(NAMESPACE) \
     -f deploy/values.yaml \
     -f deploy/values-$(BUILD_ENV).yaml \
    --set-string image.tag=gc-${SHA}

## helm upgrade
helm/upgrade: check-env
    helm upgrade $(RELEASE_NAME) library/bam-simple-webapp \
     --namespace $(NAMESPACE) \
     --install \
     --wait \
     --kube-context=$(KUBE_CONTEXT) \
     --namespace $(NAMESPACE) \
     -f deploy/values.yaml \
     -f deploy/values-$(BUILD_ENV).yaml \
    --set-string image.tag=gc-${SHA}

## docker/build, docker/push, helm/upgrade
app/regenerate: check-env docker/build docker/push helm/upgrade
    echo 'rebuilt and upgraded!'

## helm template
helm/template:
    helm template library/bam-simple-webapp \ 
     --kube-context=$(KUBE_CONTEXT) \
     --namespace $(NAMESPACE) \
     -f deploy/values.yaml \
     -f deploy/values-$(BUILD_ENV).yaml \
    --set-string image.tag=gc-${SHA}
    --set-string http.host=sde-${GITREF}.bamfunds.cloud

## helm delete
helm/delete: check-env
    helm delete $(RELEASE_NAME) --namespace $(NAMESPACE) --kube-context=$(KUBE_CONTEXT)

## release a review app
review-app: check-env docker/build docker/push
    #TODO add a self-desctruct/cleanup task
    helm upgrade $(REVIEW_APP_NAME) library/bam-simple-webapp \
     --namespace $(NAMESPACE) \
     --install \
     --wait \
     --kube-context=$(KUBE_CONTEXT) \
     --namespace $(NAMESPACE) \
     -f deploy/values.yaml \
     -f deploy/values-$(BUILD_ENV).yaml \
    --set-string image.tag=gc-${SHA}
    --set-string http.host=sde-${GITREF}.bamfunds.cloud