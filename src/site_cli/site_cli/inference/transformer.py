import json
import os

import botocore
import click
from pkg_resources import resource_filename
from sagemaker.tensorflow import TensorFlowModel

from ..utils import HiddenPrints


class Transformer:
    def __init__(self, job_id, phase, sagemaker_session):
        self.job_id = job_id
        self.phase = phase
        self.job_name = f"{job_id}-{phase}-inference"
        self.sagemaker_session = sagemaker_session

        with open(
            os.path.abspath(
                resource_filename(
                    "site_cli.inference.package_data.config", "global.json"
                )
            ),
            "r",
        ) as f:
            self.config = json.load(f)
        with open(
            os.path.abspath(
                resource_filename(
                    "site_cli.inference.package_data.config", f"{phase}_transform.json"
                )
            ),
            "r",
        ) as f:
            self.config.update(json.load(f))

        self.model = TensorFlowModel(
            sagemaker_session=sagemaker_session,
            model_data=self.config["model_s3"],
            role=self.config["sagemaker_role"],
            framework_version=self.config["model_framework_version"],
        )

    def run(self):
        transformer = self.model.transformer(
            instance_count=self.config["instance_count"],
            instance_type=self.config["instance_type"],
            max_concurrent_transforms=self.config["max_concurrent_transforms"],
            max_payload=self.config["max_payload"],
            output_path=(
                f"{self.config['inference_s3']}/{self.job_id}/{self.phase}_output_crops"
            ),
        )
        if self.status == "Completed":
            click.echo(f"Completed job {self.job_id}-{self.phase}-inference found")
            return
        if self.status == "InProgress":
            click.echo(f"InProgress job {self.job_id}-{self.phase}-inference found")
            return
        if self.status not in ("Completed", "InProgress", "NotExist"):
            self.job_name = f"{self.job_name}r"
            self.run()
            return
        with HiddenPrints():
            transformer.transform(
                f"{self.config['inference_s3']}/{self.job_id}/{self.phase}_input_crops",
                content_type="application/x-image",
                logs=False,
                wait=False,
                job_name=self.job_name,
            )

    @property
    def status(self):
        try:
            return self.sagemaker_session.describe_transform_job(self.job_name)[
                "TransformJobStatus"
            ]
        except botocore.errorfactory.ClientError:
            return "NotExist"

    def billed_time(self):
        desc = self.sagemaker_session.describe_transform_job(self.job_name)
        return round(
            (desc["TransformEndTime"] - desc["TransformStartTime"]).total_seconds()
        )

    def elapsed_time(self):
        desc = self.sagemaker_session.describe_transform_job(self.job_name)
        return round((desc["TransformEndTime"] - desc["CreationTime"]).total_seconds())
