from __future__ import print_function

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam

from n2v_simple_models import get_n2v_model_dropout, bce_with_mask, mae_with_mask, mse_with_mask


def run_n2v_simple_training(img_patch_size, save_npy_file_name, proj_name, batch_size, save_model_period, num_iters,
                            loss_mode, blind_pixel_ratio, dropout_rate, filter_size, blind_spot_radius_h,
                            blind_spot_radius_v):
    if save_npy_file_name[-4:] != '.npy':
        save_npy_file_name = save_npy_file_name + '.npy'
    plt.switch_backend('agg')

    img_rows = img_patch_size
    img_cols = img_patch_size

    save_model_path = 'save/models/'
    save_img_path = 'save/imgs/'
    save_image_period = 1

    data = np.load('processed_data/' + save_npy_file_name)

    num_train_sample = data.shape[0]
    data_rows = data.shape[1]
    data_cols = data.shape[2]

    def linear_lr_schedule(current_iter, iter_start_decay, tot_iter, initial_lr, end_lr):
        if current_iter <= iter_start_decay:
            return initial_lr
        else:
            slope = (initial_lr - end_lr) / (tot_iter - iter_start_decay)
            return initial_lr - slope * (current_iter - iter_start_decay)

    # training settings
    initial_lr = 4e-4
    iter_start_decay = 1
    end_lr = 1e-7

    # training
    optimizer = Adam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model = get_n2v_model_dropout(img_rows, img_cols, dropout_rate=dropout_rate, filter_size=filter_size)
    if loss_mode == 'bce':
        model.compile(loss=bce_with_mask, optimizer=optimizer)
    elif loss_mode == 'mae':
        model.compile(loss=mae_with_mask, optimizer=optimizer)
    elif loss_mode == 'mse':
        model.compile(loss=mse_with_mask, optimizer=optimizer)

    model.summary()

    num_blind_pix = int(blind_pixel_ratio * img_rows * img_rows)
    samples_per_iter = 256
    x_train = np.ndarray((samples_per_iter, img_rows, img_cols, 2))
    y_train = np.ndarray((samples_per_iter, img_rows, img_cols, 2))

    model.save_weights(save_model_path + proj_name + '_initial_model.hdf5')

    for iteration in range(0, num_iters, 1):
        lr = linear_lr_schedule(current_iter=iteration, iter_start_decay=iter_start_decay, tot_iter=num_iters,
                                initial_lr=initial_lr, end_lr=end_lr)
        K.set_value(model.optimizer.lr, lr)

        for i in range(samples_per_iter):
            row_pos = np.random.randint(0, data_rows - img_rows - 1)
            col_pos = np.random.randint(0, data_cols - img_cols - 1)
            slice_pos = np.random.randint(0, num_train_sample)

            tmp_img = data[slice_pos, row_pos:row_pos + img_rows, col_pos:col_pos + img_cols, :]
            tmp_modified_img = np.copy(tmp_img)
            tmp_mask = np.zeros(tmp_img.shape)

            for j in range(num_blind_pix):
                rr = np.random.randint(blind_spot_radius_v - 1, img_rows - blind_spot_radius_v - 1, size=1)
                cc = np.random.randint(blind_spot_radius_h - 1, img_cols - blind_spot_radius_h - 1, size=1)
                tmp_mask[rr, cc] = 1

                for spot_i in range(1 - blind_spot_radius_v, blind_spot_radius_v, 1):
                    for spot_j in range(1 - blind_spot_radius_h, blind_spot_radius_h, 1):
                        rrr = np.random.randint(0, img_rows - 1, size=1)
                        ccc = np.random.randint(0, img_cols - 1, size=1)
                        tmp_modified_img[rr + spot_i, cc + spot_j] = tmp_img[rrr, ccc]

            x_train[i, :, :, 0] = tmp_modified_img.squeeze()
            x_train[i, :, :, 1] = tmp_mask.squeeze()
            y_train[i, :, :, 0] = tmp_img.squeeze()
            y_train[i, :, :, 1] = tmp_mask.squeeze()
        print('Iteration: ' + str(iteration + 1) + ', training data generated. Start current iteration...')

        model.fit(x=x_train, y=y_train, epochs=1, batch_size=batch_size, shuffle=True, verbose=2)
        np.random.shuffle(x_train)
        x_test = x_train[0:5, :]
        tmp_pred = model.predict(x_test, batch_size=batch_size, verbose=0)

        if (iteration + 1) % save_model_period == 0:
            model.save_weights(save_model_path + proj_name + '_model_' + str(iteration + 1) + '.hdf5')

        if (iteration + 1) % save_image_period == 0:
            tmp1 = np.concatenate(
                (x_test[0, :, :, 0].squeeze(), x_test[1, :, :, 0].squeeze(), x_test[2, :, :, 0].squeeze(),
                 x_test[3, :, :, 0].squeeze(), x_test[4, :, :, 0].squeeze()), axis=1)
            tmp2 = np.concatenate((tmp_pred[0, :, :, 0].squeeze(), tmp_pred[1, :, :, 0].squeeze(),
                                   tmp_pred[2, :, :, 0].squeeze(), tmp_pred[3, :, :, 0].squeeze(),
                                   tmp_pred[4, :, :, 0].squeeze()), axis=1)
            tmp3 = np.concatenate((tmp1, tmp2), axis=0).squeeze()
            tmp_rows = tmp3.shape[0]
            tmp_cols = tmp3.shape[1]
            tmp3 = tmp3.reshape((tmp_rows, tmp_cols, 1))
            tmp4 = np.concatenate((np.zeros((tmp_rows, tmp_cols, 1)), tmp3, np.zeros((tmp_rows, tmp_cols, 1))), axis=2)
            plt.imsave(save_img_path + proj_name + '_' + str(iteration + 1) + '.png', tmp4, vmin=0, vmax=1)

    model.save_weights(save_model_path + proj_name + '_final_model.hdf5')
