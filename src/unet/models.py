from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Input,
    Lambda,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    add,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file


def unet(input_size, n_classes, encoder="resnet50", pretrained=True):
    """
    Create a U-net model

    Parameters
    ----------
    input_size : tuple
        Size of input images. 2-tuple of integers (width, height)

    n_classes : int
        Number of classes, including "others" class

    encoder : str, optional
        Type of encoder. Default: resnet50

    pretrained : bool, optional
        Whether to load pretrained weights. Default: True

    Returns
    -------
    TF model
        A U-net model

    """
    if encoder == "vanilla_cnn":
        img_input, encoder_blocks = vanilla_cnn_encoder(input_size)
    elif encoder == "vgg":
        img_input, encoder_blocks = vgg_encoder(input_size, pretrained)
    elif encoder == "resnet50":
        img_input, encoder_blocks = resnet50_encoder(input_size, pretrained)
    else:
        raise ValueError(f"Unknown encoder {encoder}.")

    [f1, f2, f3, f4, f5] = encoder_blocks
    o = f5

    # Block 1
    o = (ZeroPadding2D((1, 1), data_format="channels_last"))(o)
    o = (
        Conv2D(
            512, (3, 3), padding="valid", activation="relu", data_format="channels_last"
        )
    )(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format="channels_last"))(o)

    # Block 2
    o = concatenate([o, f4], axis=-1)
    o = (ZeroPadding2D((1, 1), data_format="channels_last"))(o)
    o = (
        Conv2D(
            256, (3, 3), padding="valid", activation="relu", data_format="channels_last"
        )
    )(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format="channels_last"))(o)

    # Block 3
    o = concatenate([o, f3], axis=-1)
    o = (ZeroPadding2D((1, 1), data_format="channels_last"))(o)
    o = (
        Conv2D(
            128, (3, 3), padding="valid", activation="relu", data_format="channels_last"
        )
    )(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format="channels_last"))(o)

    # Block 4
    o = concatenate([o, f2], axis=-1)
    o = (ZeroPadding2D((1, 1), data_format="channels_last"))(o)
    o = (
        Conv2D(
            64, (3, 3), padding="valid", activation="relu", data_format="channels_last"
        )
    )(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format="channels_last"))(o)

    # Block 5
    o = concatenate([o, f1], axis=-1)
    o = (ZeroPadding2D((1, 1), data_format="channels_last"))(o)
    o = (
        Conv2D(
            32, (3, 3), padding="valid", activation="relu", data_format="channels_last"
        )
    )(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format="channels_last"))(o)

    # output layer
    o = Conv2D(n_classes, (3, 3), padding="same", data_format="channels_last")(o)
    o = (Activation("softmax"))(o)

    model = Model(img_input, o)

    return model


def vanilla_cnn_encoder(input_size):
    """
    Create a vanilla CNN encoder

    Parameters
    ----------
    input_size : tuple
        Size of input images. 2-tuple of integers (width, height)

    Returns
    -------
    TF layer
        Input layer of the encoder

    list
        List of encoder block layers

    """
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_size[1], input_size[0], 3))
    x = img_input

    x = (ZeroPadding2D((pad, pad), data_format="channels_last"))(x)
    x = (
        Conv2D(
            filter_size, (kernel, kernel), data_format="channels_last", padding="valid"
        )
    )(x)
    x = (BatchNormalization())(x)
    x = (Activation("relu"))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format="channels_last"))(x)
    f1 = x

    x = (ZeroPadding2D((pad, pad), data_format="channels_last"))(x)
    x = (Conv2D(128, (kernel, kernel), data_format="channels_last", padding="valid"))(x)
    x = (BatchNormalization())(x)
    x = (Activation("relu"))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format="channels_last"))(x)
    f2 = x

    x = (ZeroPadding2D((pad, pad), data_format="channels_last"))(x)
    x = (Conv2D(256, (kernel, kernel), data_format="channels_last", padding="valid"))(x)
    x = (BatchNormalization())(x)
    x = (Activation("relu"))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format="channels_last"))(x)
    f3 = x

    x = (ZeroPadding2D((pad, pad), data_format="channels_last"))(x)
    x = (Conv2D(256, (kernel, kernel), data_format="channels_last", padding="valid"))(x)
    x = (BatchNormalization())(x)
    x = (Activation("relu"))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format="channels_last"))(x)
    f4 = x

    x = (ZeroPadding2D((pad, pad), data_format="channels_last"))(x)
    x = (Conv2D(256, (kernel, kernel), data_format="channels_last", padding="valid"))(x)
    x = (BatchNormalization())(x)
    x = (Activation("relu"))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format="channels_last"))(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def vgg_encoder(input_size, pretrained=True):
    """
    Create a VGG encoder

    Parameters
    ----------
    input_size : tuple
        Size of input images. 2-tuple of integers (width, height)

    pretrained : bool, optional
        Whether to load pretrained weights. Default: True

    Returns
    -------
    TF layer
        Input layer of the encoder

    list
        List of encoder block layers

    """
    img_input = Input(shape=(input_size[1], input_size[0], 3))

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv1",
        data_format="channels_last",
    )(img_input)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv2",
        data_format="channels_last",
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block1_pool", data_format="channels_last"
    )(x)
    f1 = x

    # Block 2
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv1",
        data_format="channels_last",
    )(x)
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv2",
        data_format="channels_last",
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block2_pool", data_format="channels_last"
    )(x)
    f2 = x

    # Block 3
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv1",
        data_format="channels_last",
    )(x)
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv2",
        data_format="channels_last",
    )(x)
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv3",
        data_format="channels_last",
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block3_pool", data_format="channels_last"
    )(x)
    f3 = x

    # Block 4
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv1",
        data_format="channels_last",
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv2",
        data_format="channels_last",
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv3",
        data_format="channels_last",
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block4_pool", data_format="channels_last"
    )(x)
    f4 = x

    # Block 5
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv1",
        data_format="channels_last",
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv2",
        data_format="channels_last",
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv3",
        data_format="channels_last",
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block5_pool", data_format="channels_last"
    )(x)
    f5 = x

    if pretrained:
        pretrained_url = (
            "https://github.com/fchollet/deep-learning-models/"
            "releases/download/v0.1/"
            "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )
        VGG_Weights_path = get_file(pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(VGG_Weights_path)

    return img_input, [f1, f2, f3, f4, f5]


def one_side_pad(x):
    x = ZeroPadding2D((1, 1), data_format="channels_last")(x)

    x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(
        filters1, (1, 1), data_format="channels_last", name=conv_name_base + "2a"
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters2,
        kernel_size,
        data_format="channels_last",
        padding="same",
        name=conv_name_base + "2b",
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters3, (1, 1), data_format="channels_last", name=conv_name_base + "2c"
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    x = add([x, input_tensor])
    x = Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(
        filters1,
        (1, 1),
        data_format="channels_last",
        strides=strides,
        name=conv_name_base + "2a",
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters2,
        kernel_size,
        data_format="channels_last",
        padding="same",
        name=conv_name_base + "2b",
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters3, (1, 1), data_format="channels_last", name=conv_name_base + "2c"
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    shortcut = Conv2D(
        filters3,
        (1, 1),
        data_format="channels_last",
        strides=strides,
        name=conv_name_base + "1",
    )(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

    x = add([x, shortcut])
    x = Activation("relu")(x)
    return x


def resnet50_encoder(input_size, pretrained=True):
    """
    Create a ResNet50 encoder

    Parameters
    ----------
    input_size : tuple
        Size of input images. 2-tuple of integers (width, height)

    pretrained : bool, optional
        Whether to load pretrained weights. Default: True

    Returns
    -------
    TF layer
        Input layer of the encoder

    list
        List of encoder block layers

    """
    img_input = Input(shape=(input_size[1], input_size[0], 3))
    bn_axis = 3

    x = ZeroPadding2D((3, 3), data_format="channels_last")(img_input)
    x = Conv2D(64, (7, 7), data_format="channels_last", strides=(2, 2), name="conv1")(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), data_format="channels_last", strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")
    f5 = x

    x = AveragePooling2D((7, 7), data_format="channels_last", name="avg_pool")(x)
    # f6 = x

    if pretrained:
        pretrained_url = (
            "https://github.com/fchollet/deep-learning-models/"
            "releases/download/v0.2/"
            "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )
        VGG_Weights_path = get_file(pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(VGG_Weights_path)

    return img_input, [f1, f2, f3, f4, f5]
