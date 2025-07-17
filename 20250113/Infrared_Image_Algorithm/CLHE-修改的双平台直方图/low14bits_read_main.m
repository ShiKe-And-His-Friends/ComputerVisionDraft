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
 
input_dir = "D:\Document\多点NUC-新建WinRAR压缩文件管理器\14bits\两点\";
save_dir = "D:\Document\多点NUC-新建WinRAR压缩文件管理器\dump\";

%搜索文件夹下的文件
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".raw", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);
    
    %芯片的raw图需要裁剪
%     GrayImage = GrayImage - 16384;
     
%     GrayImage = GrayImage - min(GrayImage(:));
     GrayImage(GrayImage>16383) = 16383;
%     
    %自适应引导滤波_自适应双平台直方图均衡
    muliti_CLHE(GrayImage ,cols ,rows ,save_dir ,file_names(i) + "")
    
end


