% �����ļ��������ͼ�������
%
clc;
clear;

input_dir = "D:\Document\��ֵ���500+ͼ������\test\�ں�\";


%�����ļ����µ��ļ�
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    disp("#################  " + file_names(i) +"  #################  ");
    
    input_temp_dir = strcat(input_dir ,file_names(i) ,".png");
    input_file = char(input_temp_dir(1));
    GrayImage = imread(input_file);
    
    %����ͼƬ����
    EvaluateQuality(GrayImage);
    
end
