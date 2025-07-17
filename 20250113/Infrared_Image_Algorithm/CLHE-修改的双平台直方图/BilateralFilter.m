function filtered_image = BilateralFilter(image, window_size, sigma_d, sigma_r)
    % 获取图像的行数和列数
    [rows, cols] = size(image);
    % 初始化滤波后的图像，与原图像大小相同
    filtered_image = zeros(rows, cols);
    % 计算窗口半径（假设窗口大小为奇数）
    half_window = floor(window_size / 2);
    
    % 遍历图像的每一个像素
    for y = 1:rows
        for x = 1:cols
            % 用于存储加权和（滤波后的值）
            sum_weighted_value = 0;
            % 用于存储权重总和
            sum_weight = 0;
            % 遍历当前像素对应的滤波窗口
            for i = max(1, y - half_window):min(rows, y + half_window)
                for j = max(1, x - half_window):min(cols, x + half_window)
                    % 计算空间距离权重
                    dist = sqrt((i - y)^2 + (j - x)^2);
                    spatial_weight = exp(-dist^2 / (2 * sigma_d^2));
                    % 计算灰度相似性权重
                    intensity_diff = double(image(i, j)) - double(image(y, x));
                    range_weight = exp(-intensity_diff^2 / (2 * sigma_r^2));
                    % 计算综合权重
                    weight = spatial_weight * range_weight;
                    % 累加加权像素值和权重总和
                    sum_weighted_value = sum_weighted_value + double(image(i, j)) * weight;
                    sum_weight = sum_weight + weight;
                end
            end
            % 计算滤波后该像素的值
            filtered_image(y, x) = sum_weighted_value / sum_weight;
        end
    end
end