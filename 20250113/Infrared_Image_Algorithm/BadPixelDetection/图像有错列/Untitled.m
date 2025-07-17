%图像错列

clc;
clear;

%读取14bits的double小端raw图
cols = 640;
rows = 512;

input_dir = "C:\Users\shike\Desktop\新建文件夹\";

fid = fopen(input_dir  + "0.bin", 'r');
rawData1 = fread(fid, rows*cols, 'uint16');
fclose(fid);

fid = fopen(input_dir  + "1.bin", 'r');
rawData2 = fread(fid, rows*cols, 'uint16');
fclose(fid);

GrayImage1 = reshape(rawData1,cols ,rows);
GrayImage2 = reshape(rawData2,cols ,rows);

%芯片的raw图
GrayImage1 = GrayImage1 - 16384;
GrayImage2 = GrayImage2 - 16384;

row_means1 = mean(GrayImage1, 2);
row_means2 = mean(GrayImage2, 2);

x = 1:640; % 生成从 1 到 10 的一维数组
y1 = row_means1';
y2 = row_means2';

n= 4;
%y2 = circular_right_shift(y2, n);

% 绘制两个数组
figure; % 创建一个新的图形窗口
plot(x, y1, 'b-o', 'DisplayName', '00'); % 绘制 y1 数组，蓝色线条，圆形标记
hold on; % 保持当前图形，以便在同一图中绘制多个曲线
plot(x, y2, 'r-s', 'DisplayName', '11'); % 绘制 y2 数组，红色线条，方形标记

% 添加标题和标签
title('错位'); % 设置图形标题
xlabel('x'); % 设置 x 轴标签
ylabel('y'); % 设置 y 轴标签

% 添加图例
legend; % 显示图例，根据 DisplayName 属性显示曲线名称

% 显示网格线
grid on; % 显示网格线，方便观察数据点

result = mean_absolute_difference(y1, y2);
disp(['两个数组的平均绝对差值为: ', num2str(result)]);



function mad = mean_absolute_difference(arr1, arr2)
    % 检查两个数组的长度是否相同
    if length(arr1) ~= length(arr2)
        error('两个数组的长度必须相同');
    end
    % 计算对应元素的差值
    diff = arr1 - arr2;
    % 取差值的绝对值
    abs_diff = abs(diff);
    % 计算平均绝对差值
    mad = mean(abs_diff);
end


function shifted_array = circular_left_shift(arr, n)
    % 获取数组长度
    len = length(arr);
    % 处理 n 大于数组长度的情况
    n = mod(n, len);
    % 进行循环左移
    shifted_array = [arr(n+1:end), arr(1:n)];
end

function shifted_array = circular_right_shift(arr, n)
    % 获取数组的长度
    len = length(arr);
    % 处理 n 大于数组长度的情况
    n = mod(n, len);
    % 进行循环右移操作
    shifted_array = [arr(end - n + 1:end), arr(1:end - n)];
end

