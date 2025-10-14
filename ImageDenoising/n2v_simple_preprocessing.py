import numpy as np
from skimage import io

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def run_n2v_simple_data_preprocessing(tif_file_path, save_npy_file_name, cdf_clip_threshold):
    if save_npy_file_name[-4:] != '.npy':
        save_npy_file_name = save_npy_file_name + '.npy'
    if tif_file_path[-4:] != '.tif':
        save_npy_file_name = save_npy_file_name + '.tif'

    data = io.imread('datasets/' + tif_file_path)
    data = data.astype('float64')
    num_imgs = data.shape[0]
    img_rows = data.shape[1]
    img_cols = data.shape[2]

    if cdf_clip_threshold != 0:
        tmp_counts, tmp_edges = np.histogram(data.flatten(), 2048)
        pdf = tmp_counts / tmp_counts.sum()
        cdf = np.cumsum(pdf)
        idx = np.argmin(np.abs(cdf - (1 - cdf_clip_threshold)))
        max_val = 0.5 * (tmp_edges[idx] + tmp_edges[idx + 1])
        data[data >= max_val] = max_val
        data = normalize(data)

        tmp_counts, tmp_edges = np.histogram(data.flatten(), 2048)
        pdf = tmp_counts / tmp_counts.sum()
        cdf = np.cumsum(pdf)
        idx = np.argmin(np.abs(cdf - cdf_clip_threshold))
        min_val = 0.5 * (tmp_edges[idx] + tmp_edges[idx + 1])
        data[data <= min_val] = min_val
        data = normalize(data)
    else:
        data = normalize(data)
        print('No cdf clipping is performed')

    data = data.reshape((num_imgs, img_rows, img_cols, 1))
    #data = data * 0.8 + 0.15  # here we data preprocess
    # np.save('processed_data/' + save_npy_file_name, data)
    np.save(save_npy_file_name, data)