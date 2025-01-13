%基于高斯权重局部直方图压缩

% 读取图像（以灰度图像为例）
image = imread('D:\Document\均值相差500+图像数据\test\dump\场景4-AGF_ADPHE-融合前的基底图.png');
if size(image, 3) > 1
    image = rgb2gray(image);
end
% 确定局部窗口大小和高斯标准差
window_size = 3;
sigma = 0.5;

% 计算窗口半径
half_window = floor(window_size / 2);
[x, y] = meshgrid(-half_window:half_window, -half_window:half_window);
gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));

% 获取图像尺寸
[height, width] = size(image);
% 初始化处理后的图像
processed_image = zeros(height, width);
for y = 1:height
    for x = 1:width
        % 确定局部窗口范围
        row_start = max(1, y - half_window);
        row_end = min(height, y + half_window);
        col_start = max(1, x - half_window);
        col_end = min(width, x + half_window);
        % 提取局部窗口内的图像数据和对应的高斯权重
        local_image = image(row_start:row_end, col_start:col_end);
        local_kernel = gaussian_kernel(1:(row_end - row_start + 1), 1:(col_end - col_start + 1));
        % 构建局部加权直方图
        local_histogram = zeros(1, 256);
        for i = 1:size(local_image, 1)
            for j = 1:size(local_image, 2)
                gray_level = local_image(i, j) + 1;
                local_histogram(gray_level) = local_histogram(gray_level) + local_kernel(i, j);
            end
        end
        % 对局部直方图进行压缩（这里以简单的直方图均衡化为例）
        cumulative_histogram = cumsum(local_histogram);
        normalized_cumulative_histogram = cumulative_histogram / sum(local_histogram);
        for i = 1:size(local_image, 1)
            for j = 1:size(local_image, 2)
                old_gray_level = local_image(i, j) + 1;
                new_gray_level = round(normalized_cumulative_histogram(old_gray_level) * 255);
                processed_image(row_start + i - 1, col_start + j - 1) = new_gray_level;
            end
        end
    end
end

% 显示原始图像和处理后的图像
subplot(1, 2, 1);
imshow(image);
title('原始图像');
subplot(1, 2, 1);
imshow(uint8(processed_image));
title('基于高斯权重局部直方图压缩后的图像');