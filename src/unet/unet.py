import argparse
import json
import os

import tensorflow as tf
from train import train

from data import CropGenerator

# # comment this part. for local test only
# from tensorflow.compat.v1 import ConfigProto, InteractiveSession
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str)
    parser.add_argument(
        "--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )
    parser.add_argument(  # type:ignore
        "--hosts",  # type:ignore
        type=list,  # type:ignore
        default=json.loads(os.environ.get("SM_HOSTS")),  # type:ignore
    )  # type:ignore
    parser.add_argument(
        "--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST")
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--checkpoints-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--encoder", type=str, default="vanilla_cnn")
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=224)
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument(
        "--classes", type=str, default="asphalt,concrete,rooftop,landscape,gravel",
    )
    parser.add_argument("--train-dataset-mode", type=str, default="random")
    parser.add_argument("--crops-per-train-image", type=int, default=32)
    parser.add_argument("--crops-per-train-image-w", type=int, default=16)
    parser.add_argument("--crops-per-train-image-h", type=int, default=8)
    parser.add_argument("--crops-per-val-image-w", type=int, default=16)
    parser.add_argument("--crops-per-val-image-h", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--cache-size", type=int, default=1)

    args = parser.parse_args()

    # define training data generator
    train_data_generator = CropGenerator(
        data_dir=args.data_dir,
        classes=args.classes.split(","),
        batch_size=args.train_batch_size,
        crop_size=(args.crop_width, args.crop_height),
        model_input_size=(args.input_width, args.input_height),
        mode=args.train_dataset_mode,
        crops_per_img=(
            args.crops_per_train_image
            if args.train_dataset_mode == "random"
            else (args.crops_per_train_image_w, args.crops_per_train_image_h)
        ),
        split_mode="first",
        split_ratio=args.train_ratio,
        cache_size=args.cache_size,
    )
    # define validation data generator
    val_data_generator = CropGenerator(
        data_dir=args.data_dir,
        classes=args.classes.split(","),
        batch_size=args.val_batch_size,
        crop_size=(args.crop_width, args.crop_height),
        model_input_size=(args.input_width, args.input_height),
        mode="tile",
        crops_per_img=(args.crops_per_val_image_w, args.crops_per_val_image_h),
        split_mode="last",
        split_ratio=1.0 - args.train_ratio,
        cache_size=args.cache_size,
    )
    # train model
    my_model = train(
        train_data_generator=train_data_generator,
        val_data_generator=val_data_generator,
        encoder=args.encoder,
        pretrained=args.pretrained,
        n_classes=len(args.classes.split(",")) + 1,
        input_size=(args.input_width, args.input_height),
        epochs=args.epochs,
        checkpoints_dir=args.checkpoints_dir,
        use_multiprocessing=(False if args.cache_size > 1 else True),
    )
    # save final model
    tf.saved_model.save(my_model, os.path.join(args.sm_model_dir, "model/1/"))
