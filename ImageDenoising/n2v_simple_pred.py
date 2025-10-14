import sys
import datetime
import numpy as np
from skimage import io

from n2v_simple_models import get_n2v_model_dropout
from tifffile import imsave


if __name__ == "__main__":
    t0 = datetime.datetime.now()
    # input arguments:
    # 1: tif file name; 2: cdf clip threshold (0.005); 3:dropout rate (0.4); 4: filter size;
    # 5: trained model weight file name; 6,7: size of the padded image
    
    # NEW ADDITION: 8: Which 3D air
    
    
    tif_file_path = sys.argv[1]
    cdf_clip_threshold = float(sys.argv[2])
    dropout_rate = float(sys.argv[3])
    filter_size = int(sys.argv[4])
    model_name = sys.argv[5]
    rows_after_padding = int(sys.argv[6])
    cols_after_padding = int(sys.argv[7])
    num = sys.argv[8]

    # summary of the job
    print('*' * 64)
    print('Raw data file name: ', tif_file_path)
    print('Pre-processing CDF clip threshold: ', cdf_clip_threshold)
    print('Dropout rate: ', dropout_rate)
    print('Filter size: ', filter_size)
    print('Model name: ', model_name)
    print('Rows after padding: ', rows_after_padding)
    print('Cols after padding: ', cols_after_padding)
    print('Start data processing...')

    '''padded images??? 16 times mini'''
    model = get_n2v_model_dropout(rows_after_padding, cols_after_padding, dropout_rate, filter_size)
    model.load_weights('save/models/' + model_name)


    # here, I change the folder for data
    data = io.imread(tif_file_path + 'air_3D' + num + '.tif')
    data = data.astype('float32')

    num_imgs = data.shape[0]
    img_rows = data.shape[1]
    img_cols = data.shape[2]

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())

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

    data = data.reshape((num_imgs, img_rows, img_cols, 1))
    # here, I change the preprocessing process
    data = data * 0.8 + 0.1  # here we data preprocess

    row_pad_pre = round((rows_after_padding - img_rows) / 2.0)
    row_pad_post = round(rows_after_padding - img_rows - row_pad_pre)
    col_pad_pre = round((cols_after_padding - img_cols) / 2.0)
    col_pad_post = round(cols_after_padding - img_cols - col_pad_pre)
    data = np.pad(data, ((0, 0), (row_pad_pre, row_pad_post), (col_pad_pre, col_pad_post), (0, 1)), 'constant', constant_values=0)

    pred = model.predict(data, batch_size=1, verbose=1)

    data = []
    pred = pred[:, row_pad_pre:row_pad_pre + img_rows, col_pad_pre:col_pad_pre+img_cols, 0].squeeze()

    savename = r'U:\eng_research_nialab\Projects\LungEx\Result\denoiseResult\air_3D1.tif'
    imsave(savename, (65535.0 * pred).astype('uint16'))




    #
    #
    # # here, I change the folder for data
    # for i in range(300):
    #
    #     data = io.imread(tif_file_path + 'A' + str(i+1) + '.tif')
    #     data = data.astype('float32')
    #
    #     num_imgs = data.shape[0]
    #     img_rows = data.shape[1]
    #     img_cols = data.shape[2]
    #
    #     def normalize(x):
    #         return (x - x.min()) / (x.max() - x.min())
    #
    #     if cdf_clip_threshold != 0:
    #         tmp_counts, tmp_edges = np.histogram(data.flatten(), 2048)
    #         pdf = tmp_counts / tmp_counts.sum()
    #         cdf = np.cumsum(pdf)
    #         idx = np.argmin(np.abs(cdf - (1 - cdf_clip_threshold)))
    #         max_val = 0.5 * (tmp_edges[idx] + tmp_edges[idx + 1])
    #         data[data >= max_val] = max_val
    #         data = normalize(data)
    #
    #         tmp_counts, tmp_edges = np.histogram(data.flatten(), 2048)
    #         pdf = tmp_counts / tmp_counts.sum()
    #         cdf = np.cumsum(pdf)
    #         idx = np.argmin(np.abs(cdf - cdf_clip_threshold))
    #         min_val = 0.5 * (tmp_edges[idx] + tmp_edges[idx + 1])
    #         data[data <= min_val] = min_val
    #         data = normalize(data)
    #     else:
    #         data = normalize(data)
    #
    #     data = data.reshape((num_imgs, img_rows, img_cols, 1))
    #     # here, I change the preprocessing process
    #     data = data * 0.8 + 0.1  # here we data preprocess
    #
    #     row_pad_pre = round((rows_after_padding - img_rows) / 2.0)
    #     row_pad_post = round(rows_after_padding - img_rows - row_pad_pre)
    #     col_pad_pre = round((cols_after_padding - img_cols) / 2.0)
    #     col_pad_post = round(cols_after_padding - img_cols - col_pad_pre)
    #     data = np.pad(data, ((0, 0), (row_pad_pre, row_pad_post), (col_pad_pre, col_pad_post), (0, 1)), 'constant', constant_values=0)
    #
    #     pred = model.predict(data, batch_size=1, verbose=1)
    #
    #     data = []
    #     pred = pred[:, row_pad_pre:row_pad_pre + img_rows, col_pad_pre:col_pad_pre+img_cols, 0].squeeze()
    #
    #     savename = r'U:\eng_research_nialab\Projects\LungEx\Result\denoiseResult\Lung2D_air\A' + str(i+1) + '.tif'
    #     imsave(savename, (65535.0 * pred).astype('uint16'))
    #
    #





