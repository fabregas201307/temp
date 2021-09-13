from typing import Iterator

import boto3


def get_boto3_session(
    bucket: str, profile: str = "crayon-site", region: str = "us-east-2"
) -> boto3.session.Session:
    """creates session object

    Args:
        bucket (str): s3 bucket
        profile (str, optional): aws config profile. Defaults to "sites".
        region (str, optional): s3 bucket region. Defaults to "us-east-2".

    Returns:
        object: session object based on profile and region
    """
    boto3_session = boto3.session.Session(profile_name=profile, region_name=region)
    return boto3_session


def get_matching_s3_keys(
    bucket: str, prefix, suffix, profile: str = "crayon-site", region: str = "us-east-2"
) -> Iterator[str]:
    """ Generator function which yeilds images in s3 bucket in {prefix}
    location with {suffix} extension

    Args:
        bucket (str): [Name of the S3 bucket.]
        prefix (str): [Only fetch keys that start with this prefix (optional)]
        suffix (str): [Only fetch keys that end with this suffix (optional)]
        profile (str, optional): [aws config profile]. Defaults to "sites".
        region (str, optional): [s3 bucket region]. Defaults to "us-east-2".

    Yields:
        [generator]: file list generator
    """

    session = get_boto3_session(bucket, profile, region)
    s3 = session.client("s3")
    kwargs = {"Bucket": bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs["Prefix"] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp["Contents"]:
            key = obj["Key"]
            if key.startswith(prefix) and (
                key.endswith(suffix.lower()) or key.endswith(suffix.upper())
            ):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break
