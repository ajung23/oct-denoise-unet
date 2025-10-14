import sys
import datetime
from n2v_simple_preprocessing import run_n2v_simple_data_preprocessing
from n2v_simple_training import run_n2v_simple_training

if __name__ == "__main__":
    t0 = datetime.datetime.now()
    # input arguments:
    # 1: tif file name; 2: npy file name; 3: cdf clip threshold (0.005); 4:image patch size (128); 5: project name;
    # 6: batch size (8); 7:model saving period (128); 8: number of total iterations (512); 9: loss function mode (bce)
    # 10: blind_pixel ratio (0.03); 11: dropout rate (0.4); 12: conv filter size (3)
    # 13,14: blind spot radius along horizontal and vertical axis (use 1 for both to launch standard n2v)
    tif_file_path = sys.argv[1]
    save_npy_file_name = sys.argv[2]
    cdf_clip_threshold = float(sys.argv[3])
    img_patch_size = int(sys.argv[4])
    proj_name = sys.argv[5]
    batch_size = int(sys.argv[6])
    save_model_period = int(sys.argv[7])
    num_iters = int(sys.argv[8])
    loss_mode = sys.argv[9]
    blind_pixel_ratio = float(sys.argv[10])
    dropout_rate = float(sys.argv[11])
    filter_size = int(sys.argv[12])
    blind_spot_radius_h = int(sys.argv[13])
    blind_spot_radius_v = int(sys.argv[14])

    # summary of the job
    print('*' * 64)
    print('Raw data file name: ', tif_file_path)
    print('Saved npy file name: ', save_npy_file_name)
    print('Pre-processing CDF clip threshold: ', cdf_clip_threshold)
    print('Image patch size: ', img_patch_size)
    print('Project name: ', proj_name)
    print('Training batch size: ', batch_size)
    print('Model saving period: ', save_model_period)
    print('Number of total iterations: ', num_iters)
    print('Loss function mode: ', loss_mode)
    print('Blind pixel ratio: ', blind_pixel_ratio)
    print('Dropout rate: ', dropout_rate)
    print('filter size: ', filter_size)
    print('blind spot radius horizontal: ', blind_spot_radius_h)
    print('blind spot radius vertical: ', blind_spot_radius_v)
    print('Start data processing...')

    # run first section data processing
    run_n2v_simple_data_preprocessing(tif_file_path=tif_file_path, save_npy_file_name=save_npy_file_name,
                                      cdf_clip_threshold=cdf_clip_threshold)
    t1 = datetime.datetime.now()
    print('*' * 64)
    print('N2V: data processing is done. Elapsed time: ', t1 - t0)
    print('Start training...')

    # run second section training
    run_n2v_simple_training(img_patch_size=img_patch_size, save_npy_file_name=save_npy_file_name, proj_name=proj_name,
                            batch_size=batch_size, save_model_period=save_model_period, num_iters=num_iters,
                            loss_mode=loss_mode, blind_pixel_ratio=blind_pixel_ratio, dropout_rate=dropout_rate,
                            filter_size=filter_size, blind_spot_radius_h=blind_spot_radius_h,
                            blind_spot_radius_v=blind_spot_radius_v)
    t2 = datetime.datetime.now()
    print('*' * 64)
    print('N2V: training is done. Elapsed time: ', t2 - t1)
    print('Finishing...')
