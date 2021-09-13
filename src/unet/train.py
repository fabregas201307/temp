import os
from multiprocessing import cpu_count

from metrics import MeanIoU
from models import unet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop


def train(
    train_data_generator,
    val_data_generator,
    encoder,
    pretrained,
    n_classes,
    input_size,
    epochs,
    checkpoints_dir,
    use_multiprocessing,
):
    """
    Build and train a U-net model

    Parameters
    ----------
    train_data_generator : CropGenerator
        Data generator for training

    val_data_generator : CropGenerator
        Data generator for validation

    encoder : str
        Type of encoder

    pretrained : bool
        Whether to load pretrained weights

    n_classes : int
        Number of classes, including "others" class

    input_size : tuple
        Size of input image. 2-tuple of int (width, height)

    epochs : int
        Maximal number of epochs

    checkpoints_dir : str
        Local path at the Sagemaker instance where checkpoints are saved

    use_multiprocessing : bool
        Whether to use multiprocessing during model training

    Returns
    -------
    TF model
        A trained U-net model

    """
    if not [
        f
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".h5") and (".best." not in f)
    ]:  # if this is a fresh new training start
        model = unet(
            input_size=input_size,
            n_classes=n_classes,
            encoder=encoder,
            pretrained=pretrained,
        )
        max_epoch_number = 0
    else:  # if this is a restart of a training that was previously suspended
        files = [
            f
            for f in os.listdir(checkpoints_dir)
            if f.endswith(".h5") and (".best." not in f)
        ]
        epoch_numbers = [int(f[-6:-3]) for f in files]
        max_epoch_number = max(epoch_numbers)
        max_epoch_filename = files[epoch_numbers.index(max_epoch_number)]
        model = load_model(os.path.join(checkpoints_dir, max_epoch_filename))

    # compile model
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=[MeanIoU(num_classes=n_classes), "categorical_accuracy"],
    )

    # define callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoints_dir, encoder + ".{epoch:03d}.h5"),
            save_weights_only=False,
            monitor="val_loss",
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoints_dir, encoder + ".best.{epoch:03d}.h5"),
            save_best_only=True,
            save_weights_only=False,
            monitor="val_loss",
        ),
        EarlyStopping(monitor="val_loss", patience=20),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=0.00001),
    ]

    # train the model
    try:  # try and catch a TF bug that occasionally val metrics are not reported
        model.fit(
            train_data_generator,
            validation_data=val_data_generator,
            epochs=epochs,
            initial_epoch=max_epoch_number,
            callbacks=callbacks,
            workers=(cpu_count() if use_multiprocessing else 1),
            use_multiprocessing=use_multiprocessing,
            verbose=2,
        )
    except FileNotFoundError:  # if the TF bug is hit
        print("Hit the TF bug of missing val metrics.")
        return train(
            train_data_generator,
            val_data_generator,
            encoder,
            pretrained,
            n_classes,
            input_size,
            epochs,
            checkpoints_dir,
            use_multiprocessing,
        )
    return model
