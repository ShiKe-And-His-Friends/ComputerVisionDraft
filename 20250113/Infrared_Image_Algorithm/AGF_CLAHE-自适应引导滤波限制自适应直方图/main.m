% 2025.2.8 sk95120
%
% DDE
% input 14bits的raw图 
% output 8bits的raw图
%
% DPHE 双平台直方图
%
clc;
clear;

%读取14bits的double小端raw图
cols = 640;
rows = 512;
 
% input_dir = "C:\MATLAB_CODE\input_image\";
% save_dir = "C:\Document\均值相差500+图像数据\test\dump\";
 
% input_dir = "C:\Document\机芯图像数据-20250105\14bits\";
% save_dir = "C:\Document\机芯图像数据-20250105\dump\";

% input_dir = "C:\Users\shike\Desktop\02数据\02\";
% save_dir = "C:\Users\shike\Desktop\02数据\result2\";

input_dir = "C:\Users\shike\Desktop\03数据\03\";
save_dir = "C:\Users\shike\Desktop\03数据\result\";

% input_dir = "D:\MATLAB_CODE\短波红外上位机\20250210-2\to_dde_input\";
% save_dir = "D:\MATLAB_CODE\短波红外上位机\20250210-2\to_dde_dump3\";

%搜索文件夹下的文件
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".raw", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);
    
    %芯片的raw图需要裁剪
    %GrayImage = floor(GrayImage/4+1);
    GrayImage = GrayImage - 16383;

    correct_image = BadPixelDetection1(GrayImage ,save_dir ,file_names(i));
    GrayImage = correct_image;
    
    %自适应引导滤波_自适应双平台直方图均衡
    AGF_CLAHE(GrayImage ,cols ,rows ,save_dir ,file_names(i) + "")
    
end


