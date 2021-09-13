"""This script generates signed URL for all images in the S3 bucket, and creates
a JSON file in the same directory that could be used by LabelBox to load images.
"""

import json

import boto3
from botocore.client import Config
from botocore.exceptions import ProfileNotFound

try:
    boto3.setup_default_session(profile_name="crayon-site")
except ProfileNotFound:
    print("crayon-site profile not found. Using default aws profile.")

s3 = boto3.resource("s3")
s3_client = boto3.client("s3", "us-east-2", config=Config(signature_version="s3v4"))

# Your Bucket Name
bucket = s3.Bucket("st-crayon")

# Get the list of objects in the Bucket
s3_Bucket_iterator = bucket.objects.all()

# Generate the Signed URL for each object in the Bucket
url_list = []
for count, i in enumerate(s3_Bucket_iterator):
    url = s3_client.generate_presigned_url(
        ClientMethod="get_object", Params={"Bucket": bucket.name, "Key": i.key}
    )
    url_list.append({"externalId": i.key, "imageUrl": url})
    print(count)

# Generate JSON
with open("site-all-images-{}.json".format(count), "w") as f:
    json.dump(obj=url_list, fp=f, indent=2)
