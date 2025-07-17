% 读取图像
img = imread('D:\Document\均值相差500+图像数据\test\2025-01-18图片对比\05E短窗套自研-2-AGF_ADPHE-DDE的融合图.png');
%img = rgb2gray(img); % 如果是彩色图像，转换为灰度图像

% 设置多平台值
platforms = [20, 50, 100, 150, 200]; % 示例平台值

% 计算图像直方图
histogram = imhist(img);

% 多平台限制
num_platforms = length(platforms);
for i = 1:num_platforms + 1
    if i == 1
        start_g = 0;
        end_g = platforms(i) - 1;
    elseif i == num_platforms + 1
        start_g = platforms(i - 1);
        end_g = 255;
    else
        start_g = platforms(i - 1);
        end_g = platforms(i) - 1;
    end
    
    % 计算该区间的平台值（这里简单假设每个区间平台值相同，可根据需求调整）
    platform_value = 50; % 示例平台值
    
    % 检查该区间内像素数量是否超过平台值
    interval_hist = histogram(start_g + 1:end_g + 1);
    excess_pixels = sum(interval_hist) - platform_value * (end_g - start_g + 1);
    if excess_pixels > 0
        % 重新分配超出部分的像素
        num_gray_levels = end_g - start_g + 1;
        excess_per_level = floor(excess_pixels / num_gray_levels);
        remainder = excess_pixels - excess_per_level * num_gray_levels;
        
        for j = 1:num_gray_levels
            interval_hist(j) = platform_value + excess_per_level;
        end
        % 处理余数
        for j = 1:remainder
            interval_hist(j) = interval_hist(j) + 1;
        end
        
        histogram(start_g + 1:end_g + 1) = interval_hist;
    end
end

% 直方图均衡化
cumulative_hist = cumsum(histogram);
total_pixels = numel(img);
mapping = uint8(255 * (cumulative_hist - cumulative_hist(1)) / (total_pixels - cumulative_hist(1)));

% 灰度级映射
enhanced_img = mapping(img + 1);

% 显示原始图像和增强后的图像
subplot(1,2,1);
imshow(img);
title('原始图像');
subplot(1,2,2);
imshow(enhanced_img);
title('多平台受限直方图均衡后的图像');