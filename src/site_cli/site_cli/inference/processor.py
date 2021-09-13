import json
import os

import botocore
import click
from pkg_resources import resource_filename
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from ..utils import HiddenPrints


class Processor:
    def __init__(self, job_id, phase, mode, sagemaker_session):
        self.job_id = job_id
        self.phase = phase
        self.mode = mode
        self.job_name = f"{job_id}-{phase}-{mode}"
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
                    "site_cli.inference.package_data.config", f"{phase}_{mode}.json"
                )
            ),
            "r",
        ) as f:
            self.config.update(json.load(f))

        self.processor = ScriptProcessor(
            command=["python3"],
            image_uri=self.config["processing_repository_uri"],
            role=self.config["sagemaker_role"],
            instance_count=self.config["instance_count"],
            instance_type=self.config["instance_type"],
            volume_size_in_gb=self.config["volume_size_in_gb"],
            sagemaker_session=self.sagemaker_session,
        )

    def run(self):
        if self.status == "Completed":
            click.echo(f"Completed job {self.job_id}-{self.phase}-{self.mode} found")
            return
        if self.status == "InProgress":
            click.echo(f"InProgress job {self.job_id}-{self.phase}-{self.mode} found")
            return
        if self.status not in ("Completed", "InProgress", "NotExist"):
            self.job_name = f"{self.job_name}r"
            self.run()
            return
        if self.phase == "p1":
            if self.mode == "preprocess":
                input_s3 = [f"{self.config['inference_s3']}/{self.job_id}/input_image"]
                output_s3 = [
                    (f"{self.config['inference_s3']}/{self.job_id}/p1_input_crops")
                ]
            elif self.mode == "postprocess":
                input_s3 = [
                    (f"{self.config['inference_s3']}/{self.job_id}/p1_output_crops")
                ]
                output_s3 = [
                    (f"{self.config['inference_s3']}/{self.job_id}/p1_output_mask")
                ]
            else:
                raise ValueError("Invalid mode")
        elif self.phase == "p2":
            if self.mode == "preprocess":
                input_s3 = [f"{self.config['inference_s3']}/{self.job_id}/input_image"]
                output_s3 = [
                    (f"{self.config['inference_s3']}/{self.job_id}/p2_input_crops")
                ]
            elif self.mode == "postprocess":
                input_s3 = [
                    (f"{self.config['inference_s3']}/{self.job_id}/p2_output_crops")
                ]
                output_s3 = [
                    (f"{self.config['inference_s3']}/{self.job_id}/p2_output_mask")
                ]
            else:
                raise ValueError("Invalid mode")
        elif self.phase == "final":
            if self.mode != "report":
                raise ValueError("Invalid mode")
            input_s3 = [
                f"{self.config['inference_s3']}/{self.job_id}/input_image",
                f"{self.config['inference_s3']}/{self.job_id}/p1_output_mask",
                f"{self.config['inference_s3']}/{self.job_id}/p2_output_mask",
            ]
            output_s3 = [f"{self.config['inference_s3']}/{self.job_id}/output_report"]
        else:
            raise ValueError("Invalid phase")

        with HiddenPrints():
            self.processor.run(
                code=os.path.abspath(
                    resource_filename(
                        "site_cli.inference.package_data.scripts", f"{self.mode}.py"
                    )
                ),
                inputs=[
                    ProcessingInput(
                        source=i, destination=f"/opt/ml/processing/input/data/{k}",
                    )
                    for k, i in enumerate(input_s3)
                ],
                outputs=[
                    ProcessingOutput(
                        source=f"/opt/ml/processing/output/data/{k}",
                        destination=o,
                        s3_upload_mode="EndOfJob",
                    )
                    for k, o in enumerate(output_s3)
                ],
                arguments=sum(
                    [[k, str(v)] for k, v in self.config["processor_params"].items()],
                    [],
                ),
                logs=False,
                wait=False,
                job_name=self.job_name,
            )

    @property
    def status(self):
        try:
            return self.sagemaker_session.describe_processing_job(self.job_name)[
                "ProcessingJobStatus"
            ]
        except botocore.errorfactory.ClientError:
            return "NotExist"

    def billed_time(self):
        desc = self.sagemaker_session.describe_processing_job(self.job_name)
        return round(
            (desc["ProcessingEndTime"] - desc["ProcessingStartTime"]).total_seconds()
        )

    def elapsed_time(self):
        desc = self.sagemaker_session.describe_processing_job(self.job_name)
        return round((desc["ProcessingEndTime"] - desc["CreationTime"]).total_seconds())
