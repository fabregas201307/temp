"""
This scripts is used by SageMaker batch transform to format I/O of model endpoint calls
"""

import io
import json

import numpy as np
from PIL import Image


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == "application/x-image":
        payload = data.read()
        decoded_image = np.array(Image.open(io.BytesIO(payload)))
        return json.dumps({"instances": [decoded_image.tolist()]})
    else:
        _return_error(
            415,
            'Unsupported content type "{}"'.format(
                context.request_content_type or "Unknown"
            ),
        )


def output_handler(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode("utf-8"))
    response_content_type = context.accept_header
    prediction = response.content
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError("Error: {}, {}".format(str(code), message))
