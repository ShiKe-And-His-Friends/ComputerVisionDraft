clc;
clear;

%读取14bits的double小端raw图
cols = 1000;
rows = 683;
 
save_dir = "D:\Document\机芯图像数据-20250105\dump\";

rgbImage = imread('D:\MATLAB_CODE\CLAHE-Github\008-DIP_CLAHE_PROJ-main\example_input\mars_moon.png');

GrayImage = double(rgbImage+1);

%自适应引导滤波_自适应双平台直方图均衡
AGF_CLAHE(GrayImage ,cols ,rows ,save_dir ,"-color")


