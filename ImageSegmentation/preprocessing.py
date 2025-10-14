import math
import random

import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as sm
import scipy.io as sio
# from skimage import transform.match_histograms

def linear_normalize(tmp_input):
    tmp = tmp_input
    return (tmp - tmp.min()) / (tmp.max() - tmp.min())
def my_remove_bg(input_img):
    x = input_img.flatten()
    hist, edges = np.histogram(x, bins=1024)
    idx = np.argmax(hist)
    bg_val = 0.5 * (edges[idx] + edges[idx + 1])
    mask = ((input_img - bg_val) >= 0)
    return (input_img - bg_val)*mask
def df_depeak(df_img,ub,lb):
    df_data=df_img
    tmp_counts, tmp_edges = np.histogram(df_data, 2048)
    pdf = tmp_counts / tmp_counts.sum()
    cdf = np.cumsum(pdf)
    idx_max = np.argmin(np.abs(cdf - ub))
    idx_min = np.argmin(np.abs(cdf - lb))

    max_val = 0.5 * (tmp_edges[idx_max] + tmp_edges[idx_max + 1])
    min_val = 0.5 * (tmp_edges[idx_min] + tmp_edges[idx_min + 1])

    df_data[df_data >= max_val] = max_val
    df_data[df_data <= min_val] = min_val

    df_data = df_data / (max_val-min_val)

    return df_data
from skimage import io
from skimage import color


''' preprocessing from raw .tif files '''
src_folder = 'C:/Users/anton/OneDrive/Desktop/TianLab/Code/training/Results/'
img_rows = 512
img_cols = 1000
num_groups = 301

x_uint16 = io.imread(src_folder + 'Stack.tiff')
x_uint16 = np.expand_dims(np.transpose(x_uint16,[1,2,0]),3) # [row, col, num_samples, channels]
x_float = ((x_uint16[0:img_rows,0:img_cols,:,:]).astype('float64'))/65535
# x_float = ((x_uint16[0:img_rows,0:img_cols]).astype('float64'))/65535
x = np.transpose(x_float,[0,1,3,2]) ## in the format of [rows, cols, channels, num_samples]
# x = np.expand_dims(np.transpose(x_float,[1,0]),2) ## in the format of [rows, cols, channels, num_samples]

y_uint16 = io.imread(src_folder + 'XZ_area-Stack.tif')
# y_float = ((y_uint16[:,0:img_rows,0:img_cols]).astype('float64'))/65535
y_uint16 = np.expand_dims(np.transpose(y_uint16, [1,2,0]),3) # [row, col, num_samples, channels]
y_float = ((y_uint16[0:img_rows,0:img_cols,:,:]).astype('float64'))/65535
y = np.transpose(y_float,[0,1,3,2]) ## in the format of [rows, cols, channels, num_samples]

print(y.shape)
print(x.shape)

for i in range(x.shape[3]):
    ## hot pixel removal by morphological opening operation ##
    ## sm = skimage.morphology
    x[:, :, :, i] = sm.opening((x[:, :, :, i]), None)
    y[:, :, :, i] = sm.opening((y[:, :, :, i]), None)
    ## linear normalization between 0 and 1 ##
    x[:, :, :, i] = linear_normalize(x[:, :, :, i])
    y[:, :, :, i] = linear_normalize(y[:, :, :, i])
    ## remove constant background ##
    x[:, :, :, i] = my_remove_bg(x[:, :, :, i])
    y[:, :, :, i] = my_remove_bg(y[:, :, :, i])

''' briefly view the large-FOV data pair'''
plt.figure(figsize=[12, 8])
index=1
channel = 0
plt.subplot(1,2,1)
plt.imshow(x[:,:,channel,index])
plt.axis('off')
plt.title('input')
plt.subplot(1,2,2)
plt.imshow(y[:,:,0,index])
plt.axis('off')
plt.title('groundtruth')
plt.tight_layout()


''' generating small-FOV data pairs '''
rows = x.shape[0]
cols = x.shape[1]
out_rows = 512
out_cols = 1000
in_channels = 1
out_channels = 1
# num_train_samples = 2  ##512
# num_test_samples = 128    ##128
# num_group_train = math.ceil(num_groups*0.8)
# num_group_test = num_groups-num_group_train
num_train_samples = 300  ##512
num_test_samples = 1    ##128
num_group_train = 300
num_group_test = 1


x_train = np.ndarray((num_train_samples, out_rows, out_cols, in_channels))
x_test = np.ndarray((num_test_samples, out_rows, out_cols, in_channels))

y_train = np.ndarray((num_train_samples, out_rows, out_cols, out_channels))
y_test = np.ndarray((num_test_samples, out_rows, out_cols, out_channels))


for i in range(num_train_samples):
    ## randomly generate sample coordinates
    row_start = math.ceil(random.random() * (rows - out_rows - 1))
    col_start = math.ceil(random.random() * (cols - out_cols - 1))

    ## select large-FOV images from training group
    # k = i % num_group_train    ## Changing this because I don't like the last image as a test
    k = (i % num_group_train) + 1

    for j in range(in_channels):
        x_train[i, :, :, j] = (x[row_start:row_start + out_rows, col_start:col_start + out_cols, j, k])
    for j in range(out_channels):
        y_train[i, :, :, j] = (y[row_start:row_start + out_rows, col_start:col_start + out_cols, j, k])

for i in range(num_test_samples):
    ## randomly generate sample coordinates
    row_start = math.ceil(random.random() * (rows - out_rows - 1))
    col_start = math.ceil(random.random() * (cols - out_cols - 1))

    ## select large-FOV images from testing group
    # k = i % num_group_test + num_group_train
    k = 0   # Because I want the first image to be my test image

    # for j in range(in_channels):   # (there is only one channel, for loop useless)
    x_test[i, :, :, 0] = (x[row_start:row_start + out_rows, col_start:col_start + out_cols, 0, k])
    # for j in range(out_channels):
    y_test[i, :, :, 0] = (y[row_start:row_start + out_rows, col_start:col_start + out_cols, 0, k])

''' briefly view the small-FOV data pair'''
plt.figure(figsize=[12, 8])
index=0
channel = 0
plt.subplot(1,2,1)
plt.imshow(x_train[index, :, :, channel].squeeze())
plt.axis('off')
plt.title('input')
plt.subplot(1,2,2)
plt.imshow(y_train[index, :, :, channel].squeeze())
plt.axis('off')
plt.title('groundtruth')
plt.tight_layout()


# Cropping
# x_train.shape == (num_train_samples, rows, cols, channels)
# x * 32 = 992
x_train = x_train[:,206:334,4:996,:]
x_test = x_test[:,206:334,4:996,:]

y_train = y_train[:,206:334,4:996,:]
y_test = y_test[:,206:334,4:996,:]


# New shapes
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

''' saving all files as .npy files'''
np.save(src_folder + 'x_train.npy', y_train)
np.save(src_folder + 'x_test.npy', y_test)  # Only doing this so we have correct testing and training (FIX LATER)
# np.save(src_folder + 'x_test.npy', x_test)

np.save(src_folder + 'y_train.npy', x_train)
np.save(src_folder + 'y_test.npy', x_test)  # Only doing this so we have correct testing and training (FIX LATER)
# np.save(src_folder + 'y_test.npy', y_test)

np.save(src_folder + 'input.npy', y)
np.save(src_folder + 'ground_truth.npy', x)