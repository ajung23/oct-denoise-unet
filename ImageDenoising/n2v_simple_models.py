from __future__ import print_function

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation, \
    AveragePooling2D, Reshape, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# define conv_factory: batch normalization + ReLU + Conv2D + Dropout (optional)
def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4, bias_flag=True):
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),
                           center=bias_flag)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (5, 5), dilation_rate=(2, 2),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay),
               use_bias=bias_flag)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


# define dense block
def denseblock(x, concat_axis, nb_layers, growth_rate,
               dropout_rate=None, weight_decay=1E-4, bias_flag=True):
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay, bias_flag=bias_flag)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x


# define model rows * cols with mask
def get_n2v_model(input_rows, input_cols, filter_size):
    bias_flag = True
    inputs = Input((input_rows, input_cols, 2))
    print("inputs shape:", inputs.shape)

    # separate into two channels
    true_input = Lambda(lambda x: K.expand_dims(x[:, :, :, 0], axis=-1))(inputs)
    print("true input shape: ", true_input.shape)
    mask_layer = Lambda(lambda x: K.expand_dims(x[:, :, :, 1], axis=-1))(inputs)
    print("mask shape: ", mask_layer.shape)

    conv1 = Conv2D(64, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        true_input)
    print("conv1 shape:", conv1.shape)
    db1 = denseblock(x=conv1, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db1 shape:", db1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(128, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool1)
    print("conv2 shape:", conv2.shape)
    db2 = denseblock(x=conv2, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db2 shape:", db2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(256, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool2)
    print("conv3 shape:", conv3.shape)
    db3 = denseblock(x=conv3, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db3 shape:", db3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool3)
    print("conv4 shape:", conv4.shape)
    db4 = denseblock(x=conv4, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db4 shape:", db4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(db4)
    print("pool4 shape:", pool4.shape)

    conv5 = Conv2D(512, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool4)
    print("conv5 shape:", conv5.shape)
    db5 = denseblock(x=conv5, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db5 shape:", db5.shape)
    up5 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db5))
    print("up5 shape:", up5.shape)
    merge5 = Concatenate(axis=3)([db4, up5])
    print("merge5 shape:", merge5.shape)

    conv6 = Conv2D(512, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        merge5)
    print("conv6 shape:", conv6.shape)
    db6 = denseblock(x=conv6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db6 shape:", db6.shape)
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db6))
    print("up6 shape:", up6.shape)
    merge6 = Concatenate(axis=3)([db3, up6])
    print("merge6 shape:", merge6.shape)

    conv7 = Conv2D(256, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        merge6)
    print("conv7 shape:", conv7.shape)
    db7 = denseblock(x=conv7, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db7 shape:", db7.shape)
    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db7))
    print("up7 shape:", up7.shape)
    merge7 = Concatenate(axis=3)([db2, up7])
    print("merge7 shape:", merge7.shape)

    conv8 = Conv2D(128, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        merge7)
    print("conv8 shape:", conv8.shape)
    db8 = denseblock(x=conv8, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db8 shape:", db8.shape)
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db8))
    print("up8 shape:", up8.shape)
    merge8 = Concatenate(axis=3)([db1, up8])
    print("merge8 shape:", merge8.shape)

    conv9 = Conv2D(64, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(merge8)
    print("conv9 shape:", conv9.shape)
    db9 = denseblock(x=conv9, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5, bias_flag=bias_flag)
    print("db9 shape:", db9.shape)
    conv10 = Conv2D(32, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                    use_bias=bias_flag)(db9)
    print("conv10 shape:", conv10.shape)
    conv11 = Conv2D(1, 1, activation='sigmoid', use_bias=bias_flag)(conv10)
    print("conv11 shape:", conv11.shape)

    output_with_mask = Concatenate(axis=3)([conv11, mask_layer])
    print("output_with_mask shape:", output_with_mask.shape)

    model = Model(inputs=inputs, outputs=output_with_mask)

    return model


# define model rows * cols with mask
def get_n2v_model_dropout(input_rows, input_cols, dropout_rate, filter_size):
    bias_flag = True

    inputs = Input((input_rows, input_cols, 2))
    print("inputs shape:", inputs.shape)

    # separate into two channels
    true_input = Lambda(lambda x: K.expand_dims(x[:, :, :, 0], axis=-1))(inputs)
    print("true input shape: ", true_input.shape)
    mask_layer = Lambda(lambda x: K.expand_dims(x[:, :, :, 1], axis=-1))(inputs)
    print("mask shape: ", mask_layer.shape)

    conv1 = Conv2D(64, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        true_input)
    print("conv1 shape:", conv1.shape)
    do1 = Dropout(dropout_rate)(conv1)
    db1 = denseblock(x=do1, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db1 shape:", db1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(128, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool1)
    print("conv2 shape:", conv2.shape)
    do2 = Dropout(dropout_rate)(conv2)
    db2 = denseblock(x=do2, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db2 shape:", db2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(256, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool2)
    print("conv3 shape:", conv3.shape)
    do3 = Dropout(dropout_rate)(conv3)
    db3 = denseblock(x=do3, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db3 shape:", db3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool3)
    print("conv4 shape:", conv4.shape)
    do4 = Dropout(dropout_rate)(conv4)
    db4 = denseblock(x=do4, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db4 shape:", db4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(db4)
    print("pool4 shape:", pool4.shape)

    conv5 = Conv2D(512, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(pool4)
    print("conv5 shape:", conv5.shape)
    do5 = Dropout(dropout_rate)(conv5)
    db5 = denseblock(x=do5, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db5 shape:", db5.shape)
    up5 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db5))
    print("up5 shape:", up5.shape)
    do_up5 = Dropout(dropout_rate)(up5)
    merge5 = Concatenate(axis=3)([db4, do_up5])
    print("merge5 shape:", merge5.shape)

    conv6 = Conv2D(512, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        merge5)
    print("conv6 shape:", conv6.shape)
    do6 = Dropout(dropout_rate)(conv6)
    db6 = denseblock(x=do6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db6 shape:", db6.shape)
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db6))
    print("up6 shape:", up6.shape)
    do_up6 = Dropout(dropout_rate)(up6)
    merge6 = Concatenate(axis=3)([db3, do_up6])
    print("merge6 shape:", merge6.shape)

    conv7 = Conv2D(256, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        merge6)
    print("conv7 shape:", conv7.shape)
    do7 = Dropout(dropout_rate)(conv7)
    db7 = denseblock(x=do7, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db7 shape:", db7.shape)
    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db7))
    print("up7 shape:", up7.shape)
    do_up7 = Dropout(dropout_rate)(up7)
    merge7 = Concatenate(axis=3)([db2, do_up7])
    print("merge7 shape:", merge7.shape)

    conv8 = Conv2D(128, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(
        merge7)
    print("conv8 shape:", conv8.shape)
    do8 = Dropout(dropout_rate)(conv8)
    db8 = denseblock(x=do8, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db8 shape:", db8.shape)
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', use_bias=bias_flag)(
        UpSampling2D(size=(2, 2))(db8))
    print("up8 shape:", up8.shape)
    do_up8 = Dropout(dropout_rate)(up8)
    merge8 = Concatenate(axis=3)([db1, do_up8])
    print("merge8 shape:", merge8.shape)

    conv9 = Conv2D(64, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                   use_bias=bias_flag)(merge8)
    print("conv9 shape:", conv9.shape)
    do9 = Dropout(dropout_rate)(conv9)
    db9 = denseblock(x=do9, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=dropout_rate, bias_flag=bias_flag)
    print("db9 shape:", db9.shape)
    conv10 = Conv2D(32, filter_size, activation='relu', padding='same', kernel_initializer='he_normal',
                    use_bias=bias_flag)(db9)
    print("conv10 shape:", conv10.shape)
    do10 = Dropout(dropout_rate)(conv10)
    conv11 = Conv2D(1, 1, activation='sigmoid', use_bias=bias_flag)(do10)
    print("conv11 shape:", conv11.shape)

    output_with_mask = Concatenate(axis=3)([conv11, mask_layer])
    print("output_with_mask shape:", output_with_mask.shape)

    model = Model(inputs=inputs, outputs=output_with_mask)

    return model


# mae only on masked pixel location
def mae_with_mask(y_true, y_pred):
    gt = y_true[:, :, :, 0]
    pred = y_pred[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    loss = tf.multiply(K.abs(gt - pred), mask)
    return loss


# mse only on masked pixel location
def mse_with_mask(y_true, y_pred):
    gt = y_true[:, :, :, 0]
    pred = y_pred[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    loss = tf.multiply(K.square(gt - pred), mask)
    return loss


# bce only on masked pixel location
def bce_with_mask(y_true, y_pred):
    gt = y_true[:, :, :, 0]
    pred = y_pred[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    tmp_bce1 = tf.multiply(gt, tf.math.log(pred + 1e-7)) + tf.multiply((1 - gt), tf.math.log(1 - pred + 1e-7))
    tmp_bce2 = tf.multiply(tmp_bce1, -1)
    loss = tf.multiply(tmp_bce2, mask)
    return loss
