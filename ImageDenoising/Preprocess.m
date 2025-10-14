clear
clc
close all

I = read_tif_to_mat('E:\wanghao\OCT_denoise\n2v_simple\datasets\3D_noise_air.tif');
I1 = I(:, 1:300, :);
I2 = I(:, 301:600, :);
I3 = I(:, 601:900, :);
I4 = I(:, 701:1000, :);

filename1 = 'E:\wanghao\OCT_denoise\n2v_simple\datasets\air_3D1.tif';
write_mat_to_tif(I1, filename1);


% read tiff file to mat
function y = read_tif_to_mat(filename)

info = imfinfo(filename);
nz = size(info,1);ss

for i = 1:nz
   y(:,:,i) = imread(filename,'tif', 'Index', i, 'Info', info); 
end
end

% save 3D mat as tiff file without compression
function write_mat_to_tif(mat,filename)

% save angiogram segmentation as multi-image tiff file
imwrite(mat(:,:,1),filename,'compression','none');
[~,~,nz] = size(mat);
for i = 2:nz
   imwrite(mat(:,:,i),filename,'compression','none','writemode','append'); 
end

end





