from keras.models import Model
from keras.layers import (
    Input,
    BatchNormalization,
    Dense,
    merge,
    Conv2D,
    MaxPooling2D,
    Activation,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Dropout
)
import keras.backend as K

# Evidently this model breaks Python's default recursion limit
# This is a theano issue
import sys

sys.setrecursionlimit(10000)
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=True, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def inception_1a(input):
    conv_a1 = conv2d_bn(input, 64, 1, 1)

    conv_b1 = conv2d_bn(input, 64, 1, 1)
    conv_b2 = conv2d_bn(conv_b1, 64, 3, 3, strides=(1, 1), padding="valid")
    conv_b3 = conv2d_bn(conv_b2, 64, 3, 3, strides=(1, 1), padding="valid")

    pool_c = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="valid")

    conv_c1 = conv2d_bn(pool_c, 64, 1, 1, strides=(1, 1))

    return merge([conv_a1, conv_b3, conv_c1], mode="concat", concat_axis=1)


def inception_1b(input):
    conv_a1 = conv2d_bn(input, 64, 1, 1, strides=(1, 1))
    conv_a2 = conv2d_bn(conv_a1, 64, 3, 3, strides=(2, 2), padding="valid")


    conv_b1 = conv2d_bn(input, 64, 1, 1, strides=(1, 1), padding="valid")
    conv_b2= conv2d_bn(conv_b1, 64, 3, 3, strides=(1, 1) )
    conv_b3 = conv2d_bn(conv_b2, 64, 3, 3, strides=(2, 2))

    pool_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(input)

    return merge([conv_a2, conv_b3, pool_c], mode="concat", concat_axis=1)

def roi_layer(input):
    # TODO
    pass


def inception_2a(input):
    conv_a1 = conv2d_bn(input, 128, 1, 1, strides=(1, 1))

    conv_b1 = conv2d_bn(input, 128, 3, 3, strides=(1, 1))
    conv_b2 = conv2d_bn(conv_b1, 128, 3, 3, strides=(1, 1), padding="valid")

    # block 2
    conv_c1 = conv2d_bn(input, 128, 1, 1, strides=(1, 1))
    conv_c2 = conv2d_bn(conv_c1, 128, 3, 3, strides=(1, 1), padding="valid")
    conv_c3 = conv2d_bn(conv_c2, 128, 3, 3, strides=(1, 1), padding="valid")

    pool_c = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), border_mode="valid")(input)
    conv_d1 = conv2d_bn(pool_c, 128, 1, 1, strides=(1, 1))

    return merge([conv_a1, conv_b2, conv_c3, conv_d1], mode="concat", concat_axis=1)

def inception_2b(input):
    conv_a1 = conv2d_bn(input, 128, 1, 1, strides=(1, 1))
    conv_a2 = conv2d_bn(conv_a1, 128, 3, 3, strides=(1, 1), padding="valid")

    conv_b1 = conv2d_bn(input, 128, 1, 1, strides=(1, 1))
    conv_b2 = conv2d_bn(conv_b1, 128, 3, 3, strides=(1, 1), padding="valid")
    conv_b3 = conv2d_bn(conv_b2, 128, 3, 3, strides=(2, 2), padding="valid")

    pool_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(input)

    return merge([conv_a2, conv_b3, pool_c], mode="concat", concat_axis=1)

def inception_3a(input):
    conv_a1 = conv2d_bn(input, 256, 1, 1, strides=(1, 1) )

    conv_b1 = conv2d_bn(input, 256, 1, 1, strides=(1, 1))
    conv_b2 = conv2d_bn(conv_b1, 256, 3, 3, strides=(1, 1), padding="valid")

    conv_c1 = conv2d_bn(input, 256, 1, 1, strides=(1, 1))
    conv_c2 = conv2d_bn(conv_c1, 256, 3, 3, strides=(1, 1), padding="valid")

    pool_c = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), border_mode="valid")(input)
    conv_d1 = conv2d_bn(pool_c, 256, 1, 1, strides=(1, 1))

    return merge([conv_a1, conv_b2, conv_c2, conv_d1], mode="concat", concat_axis=1)

def inception_3b(input):
    conv_a1 = conv2d_bn(input, 256, 1, 1, strides=(1, 1) )
    conv_a2 = conv2d_bn(conv_a1, 256, 3, 3, strides=(2, 2), padding="valid")

    conv_b1 = conv2d_bn(input, 256, 1, 1, strides=(1, 1))
    conv_b2 = conv2d_bn(conv_b1, 256, 3, 3, strides=(1, 1), padding="valid")
    conv_b3 = conv2d_bn(conv_b2, 256, 3, 3, strides=(2, 2), padding="valid")

    pool_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(input)

    return merge([conv_a2, conv_b3, pool_c], mode="concat", concat_axis=1)

def base_model(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                w_decay=None):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    input = Input(shape=(3, 299, 299))

    # Conv1 ~ 6
    conv_1 = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding="valid")
    conv_2 = conv2d_bn(conv_1, 32, 3, 3, strides=(2, 2), padding="valid")
    conv_3 = conv2d_bn(conv_2, 64, 3, 3)
    pool_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(conv_3)

    # inception module
    inception_1a = inception_1a(pool_4)

    inception_1b = inception_1b(inception_1a)

    # TODO : ROI layer
    roi_layer = roi_layer(inception_1b)
    inception_2a = inception_2a(roi_layer)
    inception_2b = inception_2b(inception_2a)
    inception_3a = inception_3a(inception_2b)
    inception_3b = inception_3b(inception_3a)

    glob_pool = GlobalAveragePooling2D(pool_size=(6, 6), strides=(1, 1))(inception_3b)

    fc7 = Dense(256, activation='relu', kernel_initializer='glorot_normal')(glob_pool)
    fc7 = BatchNormalization(scale=True)(fc7)
    fc7 = Dropout(0.7)(fc7)

    predictions = Dense(16161, kernel_initializer='glorot_normal', activation='softmax')(fc7)

    model = Model(input, predictions)

    return model

if __name__ == '__main__':
    model = base_model()