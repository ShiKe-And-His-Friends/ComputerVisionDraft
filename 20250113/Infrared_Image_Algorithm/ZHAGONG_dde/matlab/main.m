% 图像处理：直方图均衡 + DDE细节增强
% 作者：AI助手
% 版本：1.0
% 日期：2025-07-10

clc;
clear;

cols = 640;
rows = 512;

%input_dir = "C:\MATLAB_CODE\input_image\";
%name = "黑体";
input_dir = "C:\Picture\新老算法对比\raw\";
name = "场景1";

fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

GrayImage2 = rot90(GrayImage,-1);
GrayImage = GrayImage2;

output_image =  DDE_Image(GrayImage);
%output_image =  DDE_GuideFilter_Image(GrayImage);