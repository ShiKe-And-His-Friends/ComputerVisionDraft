% 图像处理：直方图均衡 + DDE细节增强
% 作者：AI助手
% 版本：1.0
% 日期：2025-07-10

clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\MATLAB_CODE\input_image\";
name = "场景1";

%盲元阈值
threhold_up = 80;
threhold_down = 20;
bad_pixel_num = 0;

fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

GrayImage2 = rot90(GrayImage,-1);
GrayImage = GrayImage2;

inputImage = GrayImage;
% 读取输入图像（假设为14位灰度图像，使用16位存储）
%%inputImage = imread('场景1-ADPHE.png'); % 替换为实际图像路径
inputImage = im2double(inputImage); % 转换为双精度浮点数 [0,1]

% 参数设置（根据FPGA代码调整）
H_ALL = 640;    % 水平分辨率
V_ALL = 512;    % 垂直分辨率
MAP_Max = 165;  % 直方图映射最大值
MAP_Mid = 90;   % 中间值
MAP_Min = 89;   % 最小值
DDE_Level = 100;% DDE增强系数

% Step 1: 直方图均衡
equaledImage = histogramEqualization(inputImage, H_ALL, V_ALL, MAP_Max, MAP_Mid, MAP_Min);

% Step 2: DDE细节增强
enhancedImage = ddeEnhancement(equaledImage,inputImage, DDE_Level, H_ALL, V_ALL);

% 显示和保存结果
figure;
subplot(1,3,1); imshow(inputImage, []); title('原始图像');
subplot(1,3,2); imshow(equaledImage, []); title('直方图均衡');
subplot(1,3,3); imshow(enhancedImage, []); title('DDE增强后');
imwrite(enhancedImage, 'enhanced_image.png');

% ------------------------- 函数定义 ------------------------- %

% 函数1：直方图均衡
function equaledImage = histogramEqualization(image, H_ALL, V_ALL, MAP_Max, MAP_Mid, MAP_Min)
    [rows, cols] = size(image);
    % 初始化直方图统计
    histMap = zeros(16384, 1); % 假设8位灰度
    
    % 统计直方图
    for i = 1:rows
        for j = 1:cols
            pixel = round(image(i,j)) + 1; % 转换为8位灰度
            histMap(pixel) = histMap(pixel) + 1;
        end
    end
    
    % 计算累积分布函数 (CDF)
    cdf = cumsum(histMap);
    cdf = cdf / max(cdf); % 归一化
    
    % 映射到新范围 [MAP_Min, MAP_Max]
    equaledImage = zeros(size(image));
    for i = 1:rows
        for j = 1:cols
            pixel = round(image(i,j)) + 1;
            newPixel = MAP_Min + (MAP_Max - MAP_Min) * cdf(pixel);
            equaledImage(i,j) = newPixel / 255; % 归一化回 [0,1]
        end
    end
end

% 函数2：DDE细节增强
function enhancedImage = ddeEnhancement(image8u, raw_image, DDE_Level, H_ALL, V_ALL)
    [rows, cols] = size(raw_image);
    enhancedImage = zeros(size(raw_image));
    
    % 定义邻域窗口（5x5）
    windowSize = 5;
    halfSize = floor(windowSize / 2); % 强制转换为整数
    
    % 填充图像
    padImage = padarray(raw_image, [halfSize halfSize], 'replicate');
    padImage8u = padarray(image8u, [halfSize halfSize], 'replicate');
    
    % 遍历每个像素
    for i = halfSize + 1 : rows + halfSize
        for j = halfSize + 1 : cols + halfSize
            % 提取5x5邻域
            window = padImage(i - halfSize : i + halfSize, ...
                              j - halfSize : j + halfSize);
                          
            window8u = padImage8u(i - halfSize : i + halfSize, ...
                              j - halfSize : j + halfSize);
            
            % Step 1: 计算中心像素值
            centerPixel = window(halfSize + 1, halfSize + 1);
            centerPixel8u = window8u(halfSize + 1, halfSize + 1);
            
            % Step 2: 计算梯度（绝对差异）
            gradients = abs(window - centerPixel);
            
            % Step 3: 生成权重（基于梯度）
            weights = 1 ./ (1 + gradients); % 简化版权重
            
            % Step 4: 加权求和
            weightedSum = sum(sum(weights .* window));
            weightSum = sum(sum(weights));
            
            % Step 5: 归一化低频分量
            lowFreq = weightedSum / weightSum;
            
            % Step 6: 高频分量（中心像素 - 低频）
            highFreq = centerPixel - lowFreq;
            
            % Step 7: 应用DDE增强系数
            %enhancedPixel = lowFreq + DDE_Level / 100 * highFreq;
            
            enhancedPixel = centerPixel8u + DDE_Level / 100 * highFreq;
            
            % Step 8: 饱和处理
            enhancedPixel = max(0, min(1, enhancedPixel));
            
            % 写入结果
            enhancedImage(i - halfSize, j - halfSize) = enhancedPixel;
        end
    end
end