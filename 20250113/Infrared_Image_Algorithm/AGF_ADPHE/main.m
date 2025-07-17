% 2024.12.25 sk95120
%
% DDE
% input 14bits的raw图 
% output 8bits的raw图
%
clc;
clear;

%读取14bits的double小端raw图
cols = 640;
rows = 512;

input_dir = "D:\MATLAB_CODE\input_image\";
save_dir = "D:\Document\均值相差500+图像数据\test\dump\";

% input_dir = "D:\Document\机芯图像数据-20250105\14bits\";
% save_dir = "D:\Document\机芯图像数据-20250105\dump\";

%搜索文件夹下的文件
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".raw", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);

    %芯片的raw图需要裁剪
    GrayImage = GrayImage - 16384;
    
    %自适应引导滤波_自适应双平台直方图均衡
    AGF_ADPHE(GrayImage ,cols ,rows ,save_dir ,file_names(i) + "-AGF_ADPHE")
    
end


