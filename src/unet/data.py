import json
import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence


class CropGenerator(Sequence):
    def __init__(
        self,
        data_dir,
        classes,
        batch_size,
        crop_size,
        model_input_size=None,
        mode="random",
        crops_per_img=1,
        split_mode="first",
        split_ratio=1.0,
        shuffle=True,
        cache_size=1,
        seed=None,
    ):
        """Generator of crops from images/masks to train a network

        Parameters
        ----------
        data_dir : str
            Directory to the original images and masks. Must contains two subdirectories
            "images" and "masks".

        classes : list
            List of classes (str), excluding the "other" class.

        batch_size : int
            Batch size.

        crop_size : tuple
            Size of crops (width, height)

        model_input_size : tuple, optional
            Size of model input (width, height). If given, every crop will be rescaled
            to this size. Otherwise, the crop size is assumed to be same as model input
            size. Default: None

        mode : str, optional
            Mode of the generator, "random" or "tile". If "random", the generator cuts
            a fixed number of crops from each images randomly. If "tile", the generator
            cuts equally-spaced tiles from each images. Default: "random"

        crops_per_img : int or tuple, optional
            Number of crops cut from each images. In random mode, this is an integer.
            In tile mode, this is a 2-tuple, number of crops along horiztonal direction
            and verticle direction (i.e. the number of crops in an image is the product
            of these two numbers). Default: 1

        split_mode : str, optional
            If split the entire set of images into training and validation, whether this
            is the first x% or the last x%. Default: "first"

        split_ratio : str, optional
            Fraction of the entire set of images to be used to generate crops. Default:
            0.8

        shuffle : bool, optional
            Whether to shuffle the order of original images in each epoch. Default: True

        cache_size: int, optional
            Maximal number of images loaded into memory at the same time. Default: 1

        seed : int, optional
            Random seed. Default: None

        """
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        with open(os.path.join(self.mask_dir, "classes.json"), "r") as f:
            self.class_to_index = json.load(f)
        self.class_to_index = {
            key.lower().replace(" ", "_"): value
            for key, value in self.class_to_index.items()
        }
        self.classes = classes
        self.crop_width = crop_size[0]
        self.crop_height = crop_size[1]
        if model_input_size is None:
            self.input_width = self.crop_width
            self.input_height = self.crop_height
        else:
            self.input_width = model_input_size[0]
            self.input_height = model_input_size[1]
        all_img = os.listdir(self.img_dir)
        all_img = sorted(all_img)
        rand = np.random.RandomState(123456789)  # must always split the same way
        rand.shuffle(all_img)
        if mode not in ["random", "tile"]:
            raise ValueError("mode must be either random or tile.")
        self.mode = mode
        if split_mode == "first":
            self.img_list = np.repeat(
                all_img[: int(np.round(len(all_img) * split_ratio))],
                (
                    crops_per_img
                    if mode == "random"
                    else crops_per_img[0] * crops_per_img[1]
                ),
            )
        elif split_mode == "last":
            self.img_list = np.repeat(
                all_img[-int(np.round(len(all_img) * split_ratio)) :],
                (
                    crops_per_img
                    if mode == "random"
                    else crops_per_img[0] * crops_per_img[1]
                ),
            )
        else:
            raise ValueError("split_mode must be either first or last.")
        self.crops_per_img = crops_per_img
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rand = np.random.RandomState(seed)
        self.indexes = np.arange(len(self.img_list))
        self.cache_size = cache_size
        self.cache = dict()
        self.cache_count = dict()
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.img_list) / self.batch_size))

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle is True:
            self.rand.shuffle(self.indexes)

    def _update_cache(self, img_name):
        if img_name in self.cache.keys():
            self.cache_count[img_name] += 1
            return
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        try:
            mask = cv2.imread(
                os.path.join(self.mask_dir, f"{img_name[:-4]}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
        except OSError:
            mask = None
        if img_name not in self.cache_count.keys():
            self.cache_count[img_name] = 0
        else:
            self.cache_count[img_name] += 1
        if len(self.cache) == self.cache_size:
            self.cache.pop(
                max(
                    {k: self.cache_count[k] for k in self.cache.keys()},
                    key={k: self.cache_count[k] for k in self.cache.keys()}.get,
                )
            )
        self.cache[img_name] = (img, mask)

    def __getitem__(self, batch_idx):
        "Generate one batch of data"
        # Initialization
        X = np.zeros((self.batch_size, self.input_height, self.input_width, 3))
        Y = np.zeros(
            (
                self.batch_size,
                self.input_height,
                self.input_width,
                len(self.classes) + 1,
            )
        )

        samples_idx = self.indexes[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        imgs_batch = [self.img_list[k] for k in samples_idx]
        # Generate data
        for i, (img_name, sample_idx) in enumerate(zip(imgs_batch, samples_idx)):
            self._update_cache(img_name)
            img, mask = self.cache[img_name]
            H, W, _ = img.shape
            if self.mode == "random":
                crop_top = self.rand.choice(H - self.crop_height + 1)
                crop_left = self.rand.choice(W - self.crop_width + 1)
            else:
                crop_idx_w = (
                    sample_idx
                    % (self.crops_per_img[0] * self.crops_per_img[1])
                    % self.crops_per_img[0]
                )
                crop_idx_h = int(
                    np.floor(
                        sample_idx
                        % (self.crops_per_img[0] * self.crops_per_img[1])
                        / self.crops_per_img[0]
                    )
                )
                crop_top = int(
                    np.linspace(0, H - self.crop_height, self.crops_per_img[1])[
                        crop_idx_h
                    ]
                )
                crop_left = int(
                    np.linspace(0, W - self.crop_width, self.crops_per_img[0])[
                        crop_idx_w
                    ]
                )
            img_crop = img[
                crop_top : crop_top + self.crop_height,
                crop_left : crop_left + self.crop_width,
            ].astype(np.uint8)
            if (
                self.crop_height != self.input_height
                or self.crop_width != self.input_width
            ):
                img_crop = cv2.resize(img_crop, (self.input_width, self.input_height))
            X[i, :, :, :] = img_crop
            for j, cl in enumerate(self.classes):
                mask_crop = (
                    mask[
                        crop_top : crop_top + self.crop_height,
                        crop_left : crop_left + self.crop_width,
                    ]
                    == self.class_to_index[cl]
                ).astype(np.uint8)
                if (
                    self.crop_height != self.input_height
                    or self.crop_width != self.input_width
                ):
                    mask_crop = cv2.resize(
                        mask_crop, (self.input_width, self.input_height)
                    )
                Y[i, :, :, j] = mask_crop
            Y[i, :, :, len(self.classes)] = (1 - Y[i, :, :, :-1].sum(axis=-1)).astype(
                np.uint8
            )

        return X, Y
