% 2024.12.25 sk95120
%
% DDE
% input 14bits��rawͼ 
% output 8bits��rawͼ
%
clc;
clear;

%��ȡ14bits��doubleС��rawͼ
% cols = 640;
% rows = 512;

cols = 320;
rows = 256;

input_dir = "C:\Users\shike\Desktop\�½��ļ���\";
save_dir = "C:\Users\shike\Desktop\�½��ļ���2\";

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
    AGF_ADPHE(GrayImage ,cols ,rows ,save_dir ,file_names(i) + "-AGF_ADPHE")
    
end


