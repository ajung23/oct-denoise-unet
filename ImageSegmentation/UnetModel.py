from __future__ import print_function

from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# define a convolutional factory, used in denseblock
def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


# define a denseblock, google DenseNet for more details
def conv_factory_DO(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
    """
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x, training=True)
    return x

def denseblock(x, concat_axis, nb_layers, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory_DO(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
    return x


# Possible tweaks (Conv2D):
# filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
# kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
#              Can be a single integer to specify the same value for all spatial dimensions.
# strides=(1, 1): An integer or tuple/list of 2 integers, specifying the strides of the convolution along the 
#                 height and width. Can be a single integer to specify the same value for all spatial dimensions.
#                 Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
# padding='valid': one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in 
#                  padding with zeros evenly to the left/right or up/down of the input such that output has 
#                  the same height/width dimension as the input.
# data_format=None: A string, one of channels_last (default) or channels_first. The ordering of the dimensions 
#                   in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels)
#                   while channels_first corresponds to inputs with shape (batch_size, channels,height, width).
#                   It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. 
#                   If you never set it, then it will be channels_last.
# dilation_rate=(1, 1): an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
#                       Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying
#                       any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
# groups=1: positive integer specifying the number of groups in which the input is split along the channel axis
#           The output is the concatenation of all the groups results along the channel axis. Input channels and
#           filters must both be divisible by groups.
# activation=None: Activation function to use. If you don't specify anything, no activation is applied   (relu commonly used)
#                  (see https://www.tensorflow.org/api_docs/python/tf/keras/activations).
# use_bias=True: boolean, whether the layer uses a bias vector. (yes 99% of the time)
# kernel_initializer='glorot_uniform: Initializer for the kernel weights matrix (Defaults to 'glorot_uniform').
#                                     (see https://www.tensorflow.org/api_docs/python/tf/keras/initializers)
# bias_initializer='zeros: Initializer for the bias vector (Defaults to 'zeros)
#                          (see https://www.tensorflow.org/api_docs/python/tf/keras/initializers)
# kernel_regularizer=None: Regularizer function applied to the kernel weights matrix
#                          (see https://www.tensorflow.org/api_docs/python/tf/keras/regularizers)
# bias_regularizer=None: Regularizer function applied to the bias vector
#                        (see https://www.tensorflow.org/api_docs/python/tf/keras/regularizers)
# activity_regularizer=None: Regularizer function applied to the output of the layer (its "activation")
#                            (see https://www.tensorflow.org/api_docs/python/tf/keras/regularizers)
# kernel_constraint=None: Constraint function applied to the kernel matrix
#                         (see https://www.tensorflow.org/api_docs/python/tf/keras/constraints)
# bias_constraint=None: Constraint function applied to the bias vector
#                       (see https://www.tensorflow.org/api_docs/python/tf/keras/constraints)
#
#
#
# When using this layer as the first layer in a model, provide the keyword argument input_shape
# (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 128, 3)
# for 128x128 RGB pictures in data_format="channels_last". You can use None when a dimension has variable size.


# U-Net modified with denseblocks
def unet_with_denseblock(input_shape, nb_output_channels):

    ##########################################################################
    # Here I want to define some of the parameters that will be the same
    # throughout the code:
    my_activation = 'relu'
    my_padding = 'same'
    my_kernel_initializer = 'he_normal'



    # Input() is used to instantiate a Keras tensor.
    input1 = Input(input_shape)
    
    # tf.keras.Input(
    #     shape=None, batch_size=None, name=None, dtype=None, sparse=None, tensor=None,
    #     ragged=None, type_spec=None, **kwargs
    # )
    # shape: A shape tuple (integers), not including the batch size. For instance, shape=(32,)
    #        indicates that the expected input will be batches of 32-dimensional vectors.
    #        Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.

    
    print("inputs shape:", input1.shape)

    conv1 = Conv2D(64, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(input1)
    print("conv1 shape:", conv1.shape)
    db1 = denseblock(x=conv1, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db1 shape:", db1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(128, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(pool1)
    print("conv2 shape:", conv2.shape)
    db2 = denseblock(x=conv2, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db2 shape:", db2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(256, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(pool2)
    print("conv3 shape:", conv3.shape)
    db3 = denseblock(x=conv3, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db3 shape:", db3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(pool3)
    print("conv4 shape:", conv4.shape)
    db4 = denseblock(x=conv4, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db4 shape:", db4.shape)
    drop4 = Dropout(0.5)(db4)
    print("drop4 shape:", drop4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print("pool4 shape:", pool4.shape)

    ###########################################
    #                                         #
    #                                         #
    # This is where the 'bottom of the U' is. #
    #                                         #
    #                                         #
    ###########################################

    conv5 = Conv2D(1024, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(pool4)
    print("conv5 shape:", conv5.shape)
    db5 = denseblock(x=conv5, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db5 shape:", db5.shape)
    drop5 = Dropout(0.5)(db5)
    print("drop5 shape:", drop5.shape)

    up6 = Conv2D(512, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop5))
    print("up6 shape:", up6.shape)
    merge6 = Concatenate(axis=3)([drop4, up6])
    print("merge6 shape:", merge6.shape)
    conv6 = Conv2D(512, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(merge6)
    print("conv6 shape:", conv6.shape)
    db6 = denseblock(x=conv6, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db6 shape:", db6.shape)

    up7 = Conv2D(256, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(
        UpSampling2D(size=(2, 2))(db6))
    print("up7 shape:", up7.shape)
    merge7 = Concatenate(axis=3)([db3, up7])
    print("merge7 shape:", merge7.shape)
    conv7 = Conv2D(256, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(merge7)
    print("conv7 shape:", conv7.shape)
    db7 = denseblock(x=conv7, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db7 shape:", db7.shape)

    up8 = Conv2D(128, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(
        UpSampling2D(size=(2, 2))(db7))
    print("up8 shape:", up8.shape)
    merge8 = Concatenate(axis=3)([db2, up8])
    print("merge8 shape:", merge8.shape)
    conv8 = Conv2D(128, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(merge8)
    print("conv8 shape:", conv8.shape)
    db8 = denseblock(x=conv8, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db8 shape:", db8.shape)

    up9 = Conv2D(64, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(
        UpSampling2D(size=(2, 2))(db8))
    print("up9 shape:", up9.shape)
    merge9 = Concatenate(axis=3)([db1, up9])
    print("merge9 shape:", merge9.shape)
    conv9 = Conv2D(64, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(merge9)
    print("conv9 shape:", conv9.shape)
    db9 = denseblock(x=conv9, concat_axis=3, nb_layers=3, growth_rate=24)
    print("db9 shape:", db9.shape)
    conv10 = Conv2D(16, 3, activation=my_activation, padding=my_padding, kernel_initializer=my_kernel_initializer)(db9)
    print("conv10 shape:", conv9.shape)
    conv11 = Conv2D(nb_output_channels, 1, activation='sigmoid')(conv10)
    print("conv11 shape:", conv11.shape)

    model = Model(inputs=input1, outputs=conv11)
    return model