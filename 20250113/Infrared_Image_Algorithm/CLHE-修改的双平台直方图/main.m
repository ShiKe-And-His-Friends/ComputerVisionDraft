% 2025.2.8 sk95120
%
% DDE
% input 14bits��rawͼ 
% output 8bits��rawͼ
%
% DPHE ˫ƽֱ̨��ͼ
%
clc;
clear;

%��ȡ14bits��doubleС��rawͼ
cols = 640;
rows = 512;
 
% input_dir = "D:\MATLAB_CODE\input_image\";
% save_dir = "D:\Document\��ֵ���500+ͼ������\test\dump\";
 
input_dir = "C:\Document\��оͼ������-20250105\14bits\";
save_dir = "C:\Document\��оͼ������-20250105\dump\";

% input_dir = "D:\Document\�人_������\14bits\";
% save_dir = "D:\Document\�人_������\result\";

%�����ļ����µ��ļ�
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".raw", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);
    
    %оƬ��rawͼ��Ҫ�ü�
    GrayImage = GrayImage - 16384;

    %����Ӧ�����˲�_����Ӧ˫ƽֱ̨��ͼ����
    muliti_CLHE(GrayImage ,cols ,rows ,save_dir ,file_names(i) + "-6bits")
    
end


