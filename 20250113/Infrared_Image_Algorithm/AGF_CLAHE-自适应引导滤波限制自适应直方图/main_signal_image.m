clc;
clear;

%��ȡ14bits��doubleС��rawͼ
cols = 1000;
rows = 683;
 
save_dir = "D:\Document\��оͼ������-20250105\dump\";

rgbImage = imread('D:\MATLAB_CODE\CLAHE-Github\008-DIP_CLAHE_PROJ-main\example_input\mars_moon.png');

GrayImage = double(rgbImage+1);

%����Ӧ�����˲�_����Ӧ˫ƽֱ̨��ͼ����
AGF_CLAHE(GrayImage ,cols ,rows ,save_dir ,"-color")


