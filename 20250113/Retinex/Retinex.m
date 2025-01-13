% https://blog.csdn.net/m0_46256255/article/details/133862194
% Retinex实现图像去雾
% 输入参数：
%  f——图像矩阵

% 输出参数：
%  In——结果图像

% 加载路径和所有文件
clc;clear;close all;
f = imread( 'D:\Document\均值相差500+图像数据\test\dump\场景3-AGF_ADPHE-融合前的基底图.png');
fr = f(:, :);
%数据类型归一化
mr = mat2gray(im2double(fr));
%定义alpha参数
%alpha = randi([80 100], 1)*20;
alpha = randi([80 100], 1)*100;

%定义模板大小
n = floor(min([size(f, 1) size(f, 2)])*0.5);
%计算中心
n1 = floor((n+1)/2);
for i = 1:n
    for j = 1:n
        %高斯函数
        b(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*alpha))/(pi*alpha);
    end
end
%卷积滤波
nr1 = imfilter(mr,b,'conv', 'replicate');
ur1 = log(nr1);
tr1 = log(mr);
yr1 = (tr1-ur1)/3;
%定义beta参数
beta = randi([80 100], 1)*1;
%定义模板大小
x = 32;
for i = 1:n
    for j = 1:n
        %高斯函数
        a(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*beta))/(6*pi*beta);
    end
end
%卷积滤波
nr2 = imfilter(mr,a,'conv', 'replicate');
ur2 = log(nr2);
tr2 = log(mr);
yr2 = (tr2-ur2)/3;
%定义eta参数
eta = randi([80 100], 1)*200;
for i = 1:n
    for j = 1:n
        %高斯函数
        e(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*eta))/(4*pi*eta);
    end
end
%卷积滤波
nr3 = imfilter(mr,e,'conv', 'replicate');
ur3 = log(nr3);
tr3 = log(mr);
yr3 = (tr3-ur3)/3;
dr = yr1+yr2+yr3;
cr = im2uint8(dr);
% 集成处理后的分量得到结果图像
In = cr;
%结果显示
figure;
subplot(2, 2, 1); imshow(f); title('原图像', 'FontWeight', 'Bold');
subplot(2, 2, 2); imshow(In); title('处理后的图像', 'FontWeight', 'Bold');
% 灰度化，用于计算直方图
Q  = f;
M = In;
subplot(2, 2, 3); imhist(Q, 64); title('原灰度直方图', 'FontWeight', 'Bold');
subplot(2, 2, 4); imhist(M, 64); title('处理后的灰度直方图', 'FontWeight', 'Bold');
imwrite(In, 'D:\Document\均值相差500+图像数据\test\dump\retinex.png');
