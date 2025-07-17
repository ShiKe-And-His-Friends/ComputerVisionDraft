% 评价文件夹里面的图像的质量
%
clc;
clear;

input_dir = "D:\Document\均值相差500+图像数据\test\黑盒\";


%搜索文件夹下的文件
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    disp("#################  " + file_names(i) +"  #################  ");
    
    input_temp_dir = strcat(input_dir ,file_names(i) ,".png");
    input_file = char(input_temp_dir(1));
    GrayImage = imread(input_file);
    
    %评价图片质量
    EvaluateQuality(GrayImage);
    
end
