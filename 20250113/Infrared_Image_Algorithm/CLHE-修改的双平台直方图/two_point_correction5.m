%
% ����У��--FPGA���㷨
%
clc;
clear;
cols = 640;
rows = 512;

%����
fid = fopen("C:\Users\shike\Desktop\20250206-����У������\����20-2025-02-06-13-51-4814bit.raw", 'r');
lowRawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
lowGrayImage = reshape(lowRawData,cols ,rows);
lowGrayImage = lowGrayImage /4;

%����
fid = fopen("C:\Users\shike\Desktop\20250206-����У������\����60-2025-02-06-13-53-3114bit.raw", 'r');
highRawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
highGrayImage = reshape(highRawData,cols ,rows);
highGrayImage = highGrayImage /4;

%�����м�512x512
[row, col] = size(highGrayImage);
image = lowGrayImage(52:563,:);
mean_low_gray_image = mean(image(:));
image = highGrayImage(52:563,:);
mean_high_gray_image = mean(image(:));

%ԭʼ����
%fid = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-����У������\����ԭʼͼ��2025-02-06-13-53-5014bit.raw", 'r');
fid = fopen("C:\Users\shike\Desktop\20250206-����У������\����ԭʼͼ��2025-02-06-13-53-5014bit.raw", 'r');
RawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
inputImage = reshape(RawData,cols ,rows);
inputImage = inputImage / 4;

%������ʵֵ
k = (mean_high_gray_image - mean_low_gray_image) ./(highGrayImage - lowGrayImage);

o = mean_low_gray_image - k .* lowGrayImage;

result_image = inputImage .* k+ o;

%result_image = -result_image;

%fileID = fopen("C:\Users\shike\Desktop\20250206-����У������\result-3114bit--5.raw", 'wb');
fileID = fopen("C:\MATLAB_CODE\�̲�������λ��\20250210-2\result2\��0-16383����_��������\����ԭʼͼ��2025-02-06-13-53-5014bit--result.raw", 'wb');
fwrite(fileID, result_image, 'uint16'); 
fclose(fileID);

