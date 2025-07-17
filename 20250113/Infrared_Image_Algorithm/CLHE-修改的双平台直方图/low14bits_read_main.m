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
 
input_dir = "D:\Document\���NUC-�½�WinRARѹ���ļ�������\14bits\����\";
save_dir = "D:\Document\���NUC-�½�WinRARѹ���ļ�������\dump\";

%�����ļ����µ��ļ�
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".raw", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);
    
    %оƬ��rawͼ��Ҫ�ü�
%     GrayImage = GrayImage - 16384;
     
%     GrayImage = GrayImage - min(GrayImage(:));
     GrayImage(GrayImage>16383) = 16383;
%     
    %����Ӧ�����˲�_����Ӧ˫ƽֱ̨��ͼ����
    muliti_CLHE(GrayImage ,cols ,rows ,save_dir ,file_names(i) + "")
    
end


