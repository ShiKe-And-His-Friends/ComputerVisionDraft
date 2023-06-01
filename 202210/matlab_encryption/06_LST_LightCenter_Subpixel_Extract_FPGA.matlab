clc;
clear;
close all;

% 读取图片
str='D:\Test_Data\20230414\25\0';

% 图片尺寸
image_width = 4320;
image_height = 1088;
file_num = 1;

gauss_row = [123,239,300,239,123];
gauss_col = [122;244;292;244;122];
core_gauss_5_5=[15,29,36,29,15;29,56,70,56,29;36,70,87,70,36;29,56,70,56,29;15,29,36,29,15]; 

fpga_gauss_row_res = zeros(1 ,5);

for k = 1:file_num
    src = double(imread([str ,num2str(37077 + k) ,'.bmp']));
    src = src(1:1088,: ,1);
    [g_row ,g_col] = size(src);
    fpga_gauss_5_5 = zeros(g_row ,g_col);
    % 5x5高斯滤波
    for gi = 3 : g_row - 3 %1088
        for gj = 3 : g_col - 3 %4320
            for gk = 1 :5
                fpga_gauss_row_res(gk) = src(gi-3+gk ,gj-2) * gauss_row(1) + src(gi-3+gk ,gj-1) * gauss_row(2) + ...
                    + src(gi-3+gk ,gj) * gauss_row(3) + src(gi-3+gk ,gj+1) * gauss_row(4) + src(gi-3+gk ,gj+2) * gauss_row(5);
            end
            fpga_gauss_res = fpga_gauss_row_res * gauss_col;
            fpga_gauss_5_5(gi ,gj) = double(floor(fpga_gauss_res / 1024^2));
        end
    end
    imwrite(uint8(fpga_gauss_5_5) ,"C:\Users\s05559\Desktop\matlab c++对比\matlab对比的5x5图片.png");
end

fprintf("image process done.\n");
