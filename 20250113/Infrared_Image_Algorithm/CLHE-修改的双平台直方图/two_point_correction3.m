%
% ����У��
%

clc;
clear;
cols = 640;
rows = 512;

input_dir = "C:\MATLAB_CODE\�̲�������λ��\20250210-2\";
save_dir = "C:\MATLAB_CODE\�̲�������λ��\20250210-2\result2\��fpga����_��������\";
name = "2025-02-10-17-16-5014bit";

%����
input_temp_dir = strcat(input_dir,"0��-2025-02-10-17-06-1314bit.bin");
input_file_dir = char(input_temp_dir(1));
fid = fopen(input_file_dir, 'r');
lowRawData = fread(fid, rows*cols*10, 'uint16');
fclose(fid);
images = reshape(lowRawData,cols ,rows,10);
averageImage = mean(images, 3);
averageImage = averageImage / 4;
lowGrayImage = floor(averageImage);

%����
input_temp_dir = strcat(input_dir,"40��-2025-02-10-17-09-5714bit.bin");
input_file_dir = char(input_temp_dir(1));
fid = fopen(input_file_dir, 'r');
highRawData = fread(fid, rows*cols*10, 'uint16');
fclose(fid);
images = reshape(highRawData,cols ,rows,10);
averageImage = mean(images, 3);
averageImage = averageImage / 4;
highGrayImage = floor(averageImage);

%�����м�512x512
[row, col] = size(highGrayImage);
image = lowGrayImage(52:563,:);
mean_low_gray_image = mean(image(:));
image = highGrayImage(52:563,:);
mean_high_gray_image = mean(image(:));

%ԭʼ����
%fid = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-����У������\����ԭʼͼ��2025-02-06-13-53-5014bit.raw", 'r');
input_temp_dir = strcat(input_dir ,name ,".bin");
input_file_dir = char(input_temp_dir(1));
fid = fopen(input_file_dir, 'r');
RawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
inputImage = reshape(RawData,cols ,rows);
inputImage = inputImage / 4;

%������ʵֵ
k = (mean_high_gray_image - mean_low_gray_image) ./(highGrayImage - lowGrayImage);

%result_image = (inputImage - lowGrayImage) .* k+ lowGrayImage;

o = mean_low_gray_image - k .* lowGrayImage;

result_image = inputImage .* k+ o;

save_temp_dir = strcat(save_dir ,name ,"-result.raw");
save_temp_dir = char(save_temp_dir(1));
fileID = fopen(save_temp_dir, 'wb');
fwrite(fileID, uint16(floor(result_image)), 'uint16'); 
fclose(fileID);

