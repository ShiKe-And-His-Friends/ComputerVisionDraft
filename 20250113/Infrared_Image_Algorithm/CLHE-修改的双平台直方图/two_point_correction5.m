%
% 两点校正--FPGA的算法
%
clc;
clear;
cols = 640;
rows = 512;

%低温
fid = fopen("C:\Users\shike\Desktop\20250206-两点校正数据\低温20-2025-02-06-13-51-4814bit.raw", 'r');
lowRawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
lowGrayImage = reshape(lowRawData,cols ,rows);
lowGrayImage = lowGrayImage /4;

%高温
fid = fopen("C:\Users\shike\Desktop\20250206-两点校正数据\高温60-2025-02-06-13-53-3114bit.raw", 'r');
highRawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
highGrayImage = reshape(highRawData,cols ,rows);
highGrayImage = highGrayImage /4;

%计算中间512x512
[row, col] = size(highGrayImage);
image = lowGrayImage(52:563,:);
mean_low_gray_image = mean(image(:));
image = highGrayImage(52:563,:);
mean_high_gray_image = mean(image(:));

%原始场景
%fid = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-两点校正数据\场景原始图像2025-02-06-13-53-5014bit.raw", 'r');
fid = fopen("C:\Users\shike\Desktop\20250206-两点校正数据\场景原始图像2025-02-06-13-53-5014bit.raw", 'r');
RawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
inputImage = reshape(RawData,cols ,rows);
inputImage = inputImage / 4;

%设置真实值
k = (mean_high_gray_image - mean_low_gray_image) ./(highGrayImage - lowGrayImage);

o = mean_low_gray_image - k .* lowGrayImage;

result_image = inputImage .* k+ o;

%result_image = -result_image;

%fileID = fopen("C:\Users\shike\Desktop\20250206-两点校正数据\result-3114bit--5.raw", 'wb');
fileID = fopen("C:\MATLAB_CODE\短波红外上位机\20250210-2\result2\用0-16383方法_仿真两点\场景原始图像2025-02-06-13-53-5014bit--result.raw", 'wb');
fwrite(fileID, result_image, 'uint16'); 
fclose(fileID);

