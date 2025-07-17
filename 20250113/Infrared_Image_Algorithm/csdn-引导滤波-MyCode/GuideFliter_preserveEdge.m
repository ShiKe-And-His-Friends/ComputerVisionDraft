%I = double(imread('cat.bmp'));
I = double(imread('场景6-ADPHE.png'));
p = I;
windows_size = 3;
disp("windows_size = " + windows_size);
r = floor((windows_size -1)/2); % try r=2, 4, or 8
disp("radius = " + r);
eps = 0.^2; % try eps=0.1^2, 0.2^2, 0.4^2

[result_image,cof_image ,epsilon] =guidedfilter(I, p, r, eps);

input_image = uint8(round(I));
base_image = uint8(round(result_image));
detail_image = uint8(round(input_image - base_image));
subplot(221), imshow(detail_image * 64);
%subplot(122), imshow(detail_image * 128);

%细节图片用高斯滤波去噪
filter_size = [3 3];
sigma = 0.1;
gaussian_filter = fspecial('gaussian', filter_size, sigma);
filtered_img = imfilter(detail_image, gaussian_filter);
subplot(222), imshow(filtered_img * 64);

%细节融合 Ak .* Detal
conf_detail_image = uint8((I-result_image) .* cof_image);
subplot(223), imshow(conf_detail_image * 64);

%细节融合 高低增益系数
% gain_max = 1.5;
% gain_min = 1.0;
% alpha = 500;
% gain_image = alpha / epsilon * (gain_min  + (gain_max - gain_min)* cof_image);
% gain_detail_image = uint8((I-result_image) .* gain_image);


% gain_image = alpha / epsilon * (gain_min  + (gain_max - gain_min)* cof_image);
% gain_detail_image = uint8((I-result_image) .* gain_image);
my_filter_core = 1 / 64 * [4 8 4; 8 16 8; 4 8 4];
filter_image_define = imfilter((I-result_image),my_filter_core);
subplot(224), imshow(uint8(filter_image_define) * 64);

%保存raw图
fileID = fopen('D:\Document\均值相差500+图像数据\test\1.raw', 'wb');
fwrite(fileID, gain_detail_image, 'uint16'); % 根据实际数据类型调整
fclose(fileID);

