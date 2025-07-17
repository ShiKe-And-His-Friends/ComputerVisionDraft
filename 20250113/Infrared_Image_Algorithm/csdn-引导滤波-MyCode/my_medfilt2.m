function filtered_image = my_medfilt2(image, filter_size)
    [rows, cols] = size(image);
    pad_size = floor(filter_size(1)/2); % 计算边界填充大小，这里假设窗口是正方形，可扩展为矩形情况
    padded_image = padarray(image, [pad_size, pad_size], 'symmetric'); % 对图像进行边界填充
    filtered_image = zeros(rows, cols); % 初始化滤波后的图像

    for i = 1:rows
        for j = 1:cols
            % 获取以当前像素为中心的局部窗口内的数据
            window = padded_image(i:i + 2 * pad_size, j:j + 2 * pad_size);
            % 将窗口内数据展平为一维向量并排序
            sorted_window = sort(window(:));
            % 取中间值作为滤波后的值
            filtered_image(i, j) = sorted_window(ceil(length(sorted_window)/2));
        end
    end
    filtered_image = uint8(filtered_image); % 将结果转换为合适的数据类型（这里假设原图像为uint8类型）
end