% ͼ����ֱ��ͼ���� + DDEϸ����ǿ
% ���ߣ�AI����
% �汾��1.0
% ���ڣ�2025-07-10

clc;
clear;

cols = 640;
rows = 512;

%input_dir = "C:\MATLAB_CODE\input_image\";
%name = "����";
input_dir = "C:\Picture\�����㷨�Ա�\raw\";
name = "����1";

fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

GrayImage2 = rot90(GrayImage,-1);
GrayImage = GrayImage2;

output_image =  DDE_Image(GrayImage);
%output_image =  DDE_GuideFilter_Image(GrayImage);