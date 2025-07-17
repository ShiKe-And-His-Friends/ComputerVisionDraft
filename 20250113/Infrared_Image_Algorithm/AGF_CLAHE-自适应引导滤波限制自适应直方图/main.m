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
 
% input_dir = "C:\MATLAB_CODE\input_image\";
% save_dir = "C:\Document\��ֵ���500+ͼ������\test\dump\";
 
% input_dir = "C:\Document\��оͼ������-20250105\14bits\";
% save_dir = "C:\Document\��оͼ������-20250105\dump\";

% input_dir = "C:\Users\shike\Desktop\02����\02\";
% save_dir = "C:\Users\shike\Desktop\02����\result2\";

input_dir = "C:\Users\shike\Desktop\03����\03\";
save_dir = "C:\Users\shike\Desktop\03����\result\";

% input_dir = "D:\MATLAB_CODE\�̲�������λ��\20250210-2\to_dde_input\";
% save_dir = "D:\MATLAB_CODE\�̲�������λ��\20250210-2\to_dde_dump3\";

%�����ļ����µ��ļ�
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".raw", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);
    
    %оƬ��rawͼ��Ҫ�ü�
    %GrayImage = floor(GrayImage/4+1);
    GrayImage = GrayImage - 16383;

    correct_image = BadPixelDetection1(GrayImage ,save_dir ,file_names(i));
    GrayImage = correct_image;
    
    %����Ӧ�����˲�_����Ӧ˫ƽֱ̨��ͼ����
    AGF_CLAHE(GrayImage ,cols ,rows ,save_dir ,file_names(i) + "")
    
end


