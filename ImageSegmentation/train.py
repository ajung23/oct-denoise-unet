from __future__ import print_function
import keras 
import tensorflow.keras.backend as K
import math

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import numpy as np
from UnetModel import unet_with_denseblock
from skimage.filters import threshold_otsu
plt.switch_backend('agg')


# global settings
src_folder = 'C:/Users/anton/OneDrive/Desktop/TianLab/Code/training/Results/'     # save the preprocessed dataset used for training and testing
save_path = 'C:/Users/anton/OneDrive/Desktop/TianLab/Code/training/Results/'      # save results
save_img_path = 'Results/'     # save intermedia results
proj_name = 'img_predicting'
img_rows = 128
img_cols = 992
num_epochs_coarse = 1
num_epochs_fine = 1
num_iters = 500
batch_size = 1  # 1 for lab pc, 2 for scc
save_model_period = 30
save_image_period = 15
num_output_channels = 1

learning_rate_coarse = 1E-4
learning_rate_fine = 1E-5

# load data files, need to first write the dataset into numpy data
y_train = np.load(src_folder + 'y_train.npy')
y_test = np.load(src_folder + 'y_test.npy')
x_train = np.load(src_folder + 'x_train.npy')
x_test = np.load(src_folder + 'x_test.npy')

print(x_test.shape)

# training with the model: Unet + resnet+ denseblock
model = unet_with_denseblock((img_rows, img_cols, 1 ),1) #changing with input

# definethe customer loss function, not necessary, can direclty use the 'mse', 'mae' and etc
def NPCC(y_true, y_pred):
    # define customer loss function NPCC
    loss = -  K.sum(y_true*y_pred,axis=(1,2,3)) / (K.sqrt(K.mean(K.square(y_pred), axis=(1,2,3)))*K.sqrt(K.mean(K.square(y_true), axis=(1,2,3))) + 1e-10)
    return loss
def PCC(y_true, y_pred):
    # define customer loss function PCC
    return K.mean((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred))) / (K.std(y_true) * K.std(y_pred))
# calculate cross entropy
def cross_entropy(p, q):
    return -sum([p[i]*math.log2(q[i]) for i in range(len(p))])


# define the gradient descent calculation method, and compile the model
optimizer = Adam(lr=learning_rate_coarse, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer,loss='bce')
model.summary()

# training the model and plot the result
for iteration in range(num_iters):
    # change learning rate for different steps
    if iteration < 150:
        lr = 1e-4
    elif iteration < 350:
        lr = 1e-5
    elif iteration < 500:
        lr = 1e-6
    else:
        lr = 1e-7

    K.set_value(model.optimizer.lr, lr)     # manually change the learning rate with different steps

    # feed dataset to train the model, this is the key step
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs_coarse, verbose = 2, shuffle=True)

    # save results: save model weights
    if (iteration + 1) % save_model_period == 0:
        model.save_weights(save_path + proj_name + '_predictor_' + str(iteration + 1) + '.hdf5')

    # plot intermedia result
    if (iteration + 1) % save_image_period == 0:
        # for i in range(5):
        for i in range(1):
            # pred = model.predict(np.expand_dims(x_test[20 * i, :, :, :],0), batch_size=1)
            pred = model.predict(np.expand_dims(x_test[1 * i, :, :, :],0), batch_size=1)
            plt.figure(figsize=[12, 8])

            handle = plt.subplot(1, 3, 1)
            # plt.imshow(x_test[20 * i, :, :, 0].squeeze(), vmin=0, vmax=1, cmap='Greys')
            plt.imshow(x_test[1 * i, :, :, 0].squeeze(), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            handle.set_title('Input')

            handle = plt.subplot(1, 3, 2)
            plt.imshow(pred.squeeze(), vmin=0, vmax=1, cmap='Greens')
            plt.axis('off')
            handle.set_title('Predict')

            handle = plt.subplot(1, 3, 3)
            # plt.imshow(y_test[20 * i, :, :, 0].squeeze(), vmin=0, vmax=1, cmap='Greens')
            plt.imshow(y_test[1 * i, :, :, 0].squeeze(), vmin=0, vmax=1, cmap='Greens')
            plt.axis('off')
            handle.set_title('GT')

            plt.tight_layout()
            plt.savefig(save_img_path + proj_name + '_' + str(iteration + 1) + '_' + str(i + 1) + '.png')
            plt.close()
