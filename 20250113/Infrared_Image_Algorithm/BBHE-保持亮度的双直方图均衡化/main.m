% 读取图像
original_image = imread('D:\Document\均值相差500+图像数据\test\COMPARE\场景1-AGF_ADPHE-融合前的基底图.png');
% 进行BBHE处理
enhanced_image = BBHE(original_image);
enhanced_image = uint8(round(enhanced_image));
% 显示原始图像和处理后的图像
subplot(1, 2, 1);
imshow(original_image);
title('原始图像');
subplot(1, 2, 2);
imshow(enhanced_image);
title('BBHE处理后图像');

imwrite(enhanced_image, 'D:\Document\均值相差500+图像数据\test\场景.png');

function enhanced_image = BBHE(image)
% 将彩色图像转换为灰度图像（如果输入是彩色图像）
if size(image, 3) > 1
    image = rgb2gray(image);
end

% 计算图像的平均亮度（均值）
mean_brightness = mean(image(:));

% 获取图像的灰度直方图
[hist_counts, gray_levels] = imhist(image);

% 找到直方图中分割的索引位置（基于平均亮度对应的灰度级）
cumulative_hist = cumsum(hist_counts);
total_pixels = numel(image);
split_index = find(cumulative_hist >= mean_brightness / 255.0 * total_pixels, 1);

% 分割直方图为两部分
lower_hist = hist_counts(1:split_index);
upper_hist = hist_counts(split_index + 1:end);

% 计算两部分的累积分布函数（CDF）
lower_cdf = cumsum(lower_hist) / sum(lower_hist);
upper_cdf = cumsum(upper_hist) / sum(upper_hist);

% 构建映射表
mapping_table_lower = zeros(1, length(gray_levels));
mapping_table_upper = zeros(1, length(gray_levels));

for i = 1:length(gray_levels)
    if i <= split_index
        mapping_table_lower(i) = round(lower_cdf(i) * (split_index - 1));
    else
        mapping_table_upper(i) = round(upper_cdf(i - split_index) * (length(gray_levels) - split_index - 1)) + split_index;
    end
end

% 合并映射表
mapping_table = mapping_table_lower;
mapping_table(split_index + 1:end) = mapping_table_upper(split_index + 1:end);

% 根据映射表进行灰度值映射，得到增强后的图像
enhanced_image = mapping_table(image+1);

end