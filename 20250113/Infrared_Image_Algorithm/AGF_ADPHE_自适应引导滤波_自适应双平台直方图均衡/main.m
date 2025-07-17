% 2024.12.25 sk95120
%
% DDE
% input 14bits的raw图 
% output 8bits的raw图
%
clc;
clear;

%读取14bits的double小端raw图
% cols = 640;
% rows = 512;

cols = 320;
rows = 256;

input_dir = "C:\Users\shike\Desktop\新建文件夹\";
save_dir = "C:\Users\shike\Desktop\新建文件夹2\";

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


