import sys

from site_tools import export_semantic_segmentation

# Right now need to manually download the json file from the project on labelbox (this
# can be automated)
if __name__ == "__main__":
    export_name = "test_export.json"
    if len(sys.argv) > 1:
        export_name = sys.argv[1]
    export_semantic_segmentation(
        export_name,
        "labelbox/original/annotations",  # save annotations folder
        # save training images folder (if None then don't download the training images)
        "labelbox/original/images",
        verbose=False,
    )
