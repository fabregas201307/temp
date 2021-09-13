import datetime
import json
import os
import re
import time

import boto3
import botocore
import click
import sagemaker
from pkg_resources import resource_filename

from .processor import Processor
from .transformer import Transformer

S3_PATH_PATTERN = "s3://([a-z0-9-.]+)/([\x00-\x7F]+)"


@click.command()  # noqa: C901
@click.argument("image")
@click.option("--aws-profile", default=None, help="AWS profile name.")
@click.option("--restart", default=None, help="Job to restart.")
def inference(image, aws_profile, restart):
    "Run inference of an image saved in S3"

    tic = time.time()

    boto_session = boto3.Session(profile_name=aws_profile)
    s3 = boto_session.resource("s3")
    sagemaker_session = sagemaker.Session(boto_session)

    if restart is None:
        job_id = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
        click.echo(f"Inference job ID: {job_id}")

        # check input data
        if re.search(S3_PATH_PATTERN, image) is None:
            click.echo(f"Error: {image} is not a valid s3 location of an object.")
            return
        image_bucket = re.search(S3_PATH_PATTERN, image)[1]
        image_key = re.search(S3_PATH_PATTERN, image)[2]

        try:
            s3.Object(image_bucket, image_key).load()
        except botocore.exceptions.ClientError:
            click.echo(f"Error: {image} is not found.")
            return

        # copy input data to inference s3
        click.echo("Copy input image")
        with open(
            os.path.abspath(
                resource_filename(
                    "site_cli.inference.package_data.config", "global.json"
                )
            ),
            "r",
        ) as f:
            config = json.load(f)
        inference_s3 = config["inference_s3"]
        inference_bucket = re.search(S3_PATH_PATTERN, inference_s3)[1]
        inference_prefix = re.search(S3_PATH_PATTERN, inference_s3)[2]
        s3.Bucket(inference_bucket).copy(
            {"Bucket": image_bucket, "Key": image_key},
            (
                f"{inference_prefix}/{job_id}/input_image/"
                f"image{image_key[image_key.rfind('.'):]}"
            ),
        )
    else:
        job_id = restart
        click.echo(f"Inference job ID: {job_id}")

        # check input data
        with open(
            os.path.abspath(
                resource_filename(
                    "site_cli.inference.package_data.config", "global.json"
                )
            ),
            "r",
        ) as f:
            config = json.load(f)
        inference_s3 = config["inference_s3"]
        inference_bucket = re.search(S3_PATH_PATTERN, inference_s3)[1]
        inference_prefix = re.search(S3_PATH_PATTERN, inference_s3)[2]

        if (
            len(
                list(
                    s3.Bucket(inference_bucket).objects.filter(
                        Prefix=f"{inference_prefix}/{job_id}/input_image/"
                    )
                )
            )
            == 1
        ):
            click.echo(f"Found previous job {job_id}")
        else:
            click.echo(f"Error: cannot find previous job {job_id}")
            return

    # start P1 preprocessing
    pre_processor_p1 = Processor(
        job_id=job_id,
        phase="p1",
        mode="preprocess",
        sagemaker_session=sagemaker_session,
    )
    click.echo("P1 preprocessing started")
    pre_processor_p1.run()

    # start P2 preprocessing
    pre_processor_p2 = Processor(
        job_id=job_id,
        phase="p2",
        mode="preprocess",
        sagemaker_session=sagemaker_session,
    )
    click.echo("P2 preprocessing started")
    pre_processor_p2.run()

    # listen to P1/P2 preprocessing status and start batch transform
    p1_preprocess_completed = False
    p2_preprocess_completed = False
    while True:
        print(".", end="", flush=True)
        if (pre_processor_p1.status == "Completed") and (not p1_preprocess_completed):
            p1_preprocess_completed = True
            click.echo("")
            click.echo(
                "P1 preprocessing completed "
                f"(elapsed time: {pre_processor_p1.elapsed_time()} seconds, "
                f"billed time: {pre_processor_p1.billed_time()} seconds)"
            )
            transformer_p1 = Transformer(
                job_id=job_id, phase="p1", sagemaker_session=sagemaker_session
            )
            click.echo("P1 batch transform inference started")
            transformer_p1.run()
            print(".", end="", flush=True)

        if (pre_processor_p2.status == "Completed") and (not p2_preprocess_completed):
            p2_preprocess_completed = True
            click.echo("")
            click.echo(
                "P2 preprocessing completed "
                f"(elapsed time: {pre_processor_p2.elapsed_time()} seconds, "
                f"billed time: {pre_processor_p2.billed_time()} seconds)"
            )
            transformer_p2 = Transformer(
                job_id=job_id, phase="p2", sagemaker_session=sagemaker_session
            )
            click.echo("P2 batch transform inference started")
            transformer_p2.run()
            print(".", end="", flush=True)

        if p1_preprocess_completed and p2_preprocess_completed:
            break

        if pre_processor_p1.status not in ("InProgress", "Completed"):
            click.echo("")
            click.echo("Error: Something wrong happened in P1 preprocessing.")
            return
        if pre_processor_p2.status not in ("InProgress", "Completed"):
            click.echo("")
            click.echo("Error: Something wrong happened in P2 preprocessing.")
            return

        time.sleep(10)

    # listen to P1/P2 batch transform status and start postprocessing
    p1_inference_completed = False
    p2_inference_completed = False
    while True:
        print(".", end="", flush=True)
        if (transformer_p1.status == "Completed") and (not p1_inference_completed):
            p1_inference_completed = True
            click.echo("")
            click.echo(
                "P1 batch transform inference completed "
                f"(elapsed time: {transformer_p1.elapsed_time()} seconds, "
                f"billed time: {transformer_p1.billed_time()} seconds)"
            )
            post_processor_p1 = Processor(
                job_id=job_id,
                phase="p1",
                mode="postprocess",
                sagemaker_session=sagemaker_session,
            )
            click.echo("P1 postprocessing started")
            post_processor_p1.run()
            print(".", end="", flush=True)
        if (transformer_p2.status == "Completed") and (not p2_inference_completed):
            p2_inference_completed = True
            click.echo("")
            click.echo(
                "P2 batch transform inference completed "
                f"(elapsed time: {transformer_p2.elapsed_time()} seconds, "
                f"billed time: {transformer_p2.billed_time()} seconds)"
            )
            post_processor_p2 = Processor(
                job_id=job_id,
                phase="p2",
                mode="postprocess",
                sagemaker_session=sagemaker_session,
            )
            click.echo("P2 postprocessing started")
            post_processor_p2.run()
            print(".", end="", flush=True)

        if p1_inference_completed and p2_inference_completed:
            break

        if transformer_p1.status not in ("InProgress", "Completed"):
            click.echo("")
            click.echo(
                "Error: Something wrong happened in P1 batch transform inference"
            )
            return
        if transformer_p2.status not in ("InProgress", "Completed"):
            click.echo("")
            click.echo(
                "Error: Something wrong happened in P2 batch transform inference"
            )
            return

        time.sleep(10)

    # listen to P1/P2 postprocessing status and start reporting
    p1_postprocess_completed = False
    p2_postprocess_completed = False
    while True:
        print(".", end="", flush=True)
        if (post_processor_p1.status == "Completed") and (not p1_postprocess_completed):
            p1_postprocess_completed = True
            click.echo("")
            click.echo(
                "P1 postprocessing completed "
                f"(elapsed time: {post_processor_p1.elapsed_time()} seconds, "
                f"billed time: {post_processor_p1.billed_time()} seconds)"
            )
        if (post_processor_p2.status == "Completed") and (not p2_postprocess_completed):
            p2_postprocess_completed = True
            click.echo("")
            click.echo(
                "P2 postprocessing completed "
                f"(elapsed time: {post_processor_p2.elapsed_time()} seconds, "
                f"billed time: {post_processor_p2.billed_time()} seconds)"
            )
        if p1_postprocess_completed and p2_postprocess_completed:
            report_processor = Processor(
                job_id=job_id,
                phase="final",
                mode="report",
                sagemaker_session=sagemaker_session,
            )
            click.echo("Reporting started")
            report_processor.run()
            break
        if post_processor_p1.status not in ("InProgress", "Completed"):
            click.echo("")
            click.echo("Error: Something wrong happened in P1 postprocessing.")
            return
        if post_processor_p2.status not in ("InProgress", "Completed"):
            click.echo("")
            click.echo("Error: Something wrong happened in P2 postprocessing.")
            return

        time.sleep(10)

    # listen to reporting status
    while True:
        print(".", end="", flush=True)
        if report_processor.status == "Completed":
            click.echo("")
            click.echo(
                "Reporting completed "
                f"(elapsed time: {report_processor.elapsed_time()} seconds, "
                f"billed time: {report_processor.billed_time()} seconds)"
            )
            break
        if report_processor.status not in ("InProgress", "Completed"):
            click.echo("")
            click.echo("Error: Something wrong happened in reporting.")
            return

        time.sleep(10)

    click.echo(
        f"Success: Inference output is saved at {inference_s3}/{job_id}/output_report"
    )

    toc = time.time()
    click.echo(f"Elapsed time: {int(toc - tic)} seconds")
