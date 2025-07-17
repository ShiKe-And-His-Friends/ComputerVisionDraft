%I = double(imread('cat.bmp'));
I = double(imread('����6-ADPHE.png'));
p = I;
windows_size = 5;
disp("windows_size = " + windows_size);
r = floor((windows_size -1)/2); % try r=2, 4, or 8
disp("radius = " + r);
eps = 10; % try eps=0.1^2, 0.2^2, 0.4^2

[result_image,~] =guidedfilter_mean(I, p, r, eps);

input_image = uint8(round(I));
base_image = uint8(round(result_image));
subplot(121), imshow(input_image);
subplot(122), imshow(base_image);

%����rawͼ
fileID = fopen('D:\Document\��ֵ���500+ͼ������\test\1.raw', 'wb');
fwrite(fileID, result_image, 'uint16'); % ����ʵ���������͵���
fclose(fileID);
