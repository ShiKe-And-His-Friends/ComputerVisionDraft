% 图像处理：直方图均衡 + DDE细节增强（14位灰度）
% 作者：AI助手
% 版本：1.1
% 日期：2025-07-10

clc;
clear;

% 图像处理：直方图均衡 + DDE细节增强（14位灰度）
% 作者：AI助手
% 版本：1.1
% 日期：2025-07-10

% 读取输入图像（假设为14位灰度图像，使用16位存储）
inputImage = imread('场景1-ADPHE.png'); % 替换为实际图像路径
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
enhancedImage = ddeEnhancement(equaledImage, DDE_Level, H_ALL, V_ALL);

% 显示和保存结果
figure;
subplot(1,3,1); imshow(inputImage, []); title('原始图像');
subplot(1,3,2); imshow(equaledImage, []); title('直方图均衡');
subplot(1,3,3); imshow(enhancedImage, []); title('DDE增强后');
imwrite(uint16(enhancedImage * 16383), 'enhanced_image_14bit.png'); % 保存为14位图像

% ------------------------- 函数定义 ------------------------- %

% 函数1：直方图均衡（适配14位灰度）
function equaledImage = histogramEqualization(image, H_ALL, V_ALL, MAP_Max, MAP_Mid, MAP_Min)
    [rows, cols] = size(image);
    % 初始化直方图统计
    histMap = zeros(16384, 1); % 14位灰度
    
    % 统计直方图
    for i = 1:rows
        for j = 1:cols
            pixel = round(image(i,j) * 16383) + 1; % 转换为14位整数
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
            pixel = round(image(i,j) * 16383) + 1;
            newPixel = MAP_Min + (MAP_Max - MAP_Min) * cdf(pixel);
            equaledImage(i,j) = newPixel / 16383; % 归一化回 [0,1]
        end
    end
end

% 函数2：DDE细节增强
function enhancedImage = ddeEnhancement(image, DDE_Level, H_ALL, V_ALL)
    [rows, cols] = size(image);
    enhancedImage = zeros(size(image));
    
    % 定义邻域窗口（5x5）
    windowSize = 5;
    padImage = padarray(image, [windowSize/2 windowSize/2], 'replicate');
    
    % 遍历每个像素
    for i = windowSize/2 + 1:rows + windowSize/2
        for j = windowSize/2 + 1:cols + windowSize/2
            % 提取5x5邻域
            window = padImage(i-windowSize/2:i+windowSize/2, j-windowSize/2:j+windowSize/2);
            
            % Step 1: 计算中心像素值
            centerPixel = window(3,3);
            
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
            enhancedPixel = lowFreq + DDE_Level/100 * highFreq;
            
            % Step 8: 饱和处理
            enhancedPixel = max(0, min(1, enhancedPixel));
            
            enhancedImage(i-windowSize/2, j-windowSize/2) = enhancedPixel;
        end
    end
end