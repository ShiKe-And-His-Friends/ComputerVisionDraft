%
% 两点校正
%
clc;
clear;
cols = 640;
rows = 512;

%低温
fid = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-两点校正数据\低温20-2025-02-06-13-51-4814bit.raw", 'r');
lowRawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
lowGrayImage = reshape(lowRawData,cols ,rows);
lowGrayImage = lowGrayImage /4;

%高温
fid = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-两点校正数据\高温60-2025-02-06-13-53-3114bit.raw", 'r');
highRawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
highGrayImage = reshape(highRawData,cols ,rows);
highGrayImage = highGrayImage /4;

%原始场景
%fid = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-两点校正数据\场景原始图像2025-02-06-13-53-5014bit.raw", 'r');
fid = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-两点校正数据\22222-2025-02-06-15-24-0414bit.raw", 'r');
RawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
inputImage = reshape(RawData,cols ,rows);
inputImage = inputImage / 4;

%设置真实值
real_high_value = 12000;
real_low_value = 500;

Th = mean(highGrayImage,2);
Tl = mean(lowGrayImage,2);

k = (real_high_value-real_low_value).*ones(size(Th)) ./ (Th -Tl);

k = repmat(k, 1, 512);

b = real_high_value.*ones(size(k)) - (real_high_value - real_low_value).*ones(size(k))./(Th -Tl) .* Th;

%结果数据
result_image = k .* inputImage+b;

result_image = -result_image;

fileID = fopen("C:\Users\zhangxiaoliang\Desktop\20250206-两点校正数据\result-3114bit--4.raw", 'wb');
fwrite(fileID, result_image, 'uint16'); 
fclose(fileID);

