function [out_image] = CLAHE_PixelMapping(num_tiles, limit, image, save_dir)
    % 打印输入参数信息
    fprintf("num_tiles =  %f %f\n", num_tiles);
    fprintf("limit = %f\n", limit);

    % 检查输入参数的有效性
    if ~is_valid_num_tiles(num_tiles)
        disp('Number of vertical and horizontal tiles must be positive');
        out_image = -4; % 错误码
        return;
    end

    if ~is_valid_limit(limit)
        disp('Limit should be in the range: [0,1]');
        out_image = -4; % 错误码
        return;
    end

    % 图像填充
    [p_image, pad_M, pad_N] = pad_image(image, num_tiles);
    [M, N] = size(p_image);
    [M_block, N_block] = calculate_block_size(p_image, num_tiles);
    block_size = [M_block, N_block];

    % 计算每个块的CDF矩阵
    cdf_matrix = make_cdf_matrix(p_image, block_size, limit);

    % 预分配输出图像和分块图像
    out_image = zeros(M, N);
    save_block_image = zeros(M, N);

    % 获取左上角和右下角块的中心像素坐标
    [xc11, yc11] = center_of_block(1, 1, block_size);
    [xcT1T2, ycT1T2] = center_of_block(num_tiles(1), num_tiles(2), block_size);

    % CLAHE处理
    for i = 1:M
        for j = 1:N
            % 获取当前像素所在块的索引和中心像素坐标
            [k, m] = block_index(i, j, block_size);
            [xc, yc] = center_of_block(k, m, block_size);

            % 根据像素位置进行不同的处理
            if is_in_frame(i, j, xc11, yc11, xcT1T2, ycT1T2)
                if is_corner_subblock(i, j, xc11, yc11, xcT1T2, ycT1T2)
                    % 角块不进行插值
                    out_image(i, j) = cdf_matrix(k, m, p_image(i, j));
                    save_block_image(i, j) = cdf_matrix(k, m, p_image(i, j));
                else
                    % 边框块进行线性插值
                    out_image(i, j) = linear_interpolation(i, j, k, m, p_image, cdf_matrix, block_size, num_tiles,yc11,ycT1T2);
                    save_block_image(i, j) = cdf_matrix(k, m, p_image(i, j));
                end
            else
                % 内部块进行双线性插值
                out_image(i, j) = bilinear_interpolation(i, j, k, m, p_image, cdf_matrix, block_size);
                save_block_image(i, j) = cdf_matrix(k, m, p_image(i, j));
            end
        end
    end

    % 保存分块图像
    save_image(save_block_image, save_dir, '-分块.raw');

    % 裁剪和转换输出图像
    [M_original, N_original] = size(image);
    out_image = uint16(floor(out_image));
    out_image = out_image(1:M_original, 1:N_original);

    % 保存插值后的图像
    save_image(out_image, save_dir, '-插值.raw');
end

function valid = is_valid_num_tiles(num_tiles)
    % 检查瓷砖数量是否为正
    valid = all(num_tiles > 0);
end

function valid = is_valid_limit(limit)
    % 检查限制参数是否在有效范围内
    valid = limit >= 0 && limit <= 1;
end

function [p_image, pad_M, pad_N] = pad_image(image, num_tiles)
    % 对图像进行填充，确保能被瓷砖完全覆盖
    [M, N] = size(image);
    pad_M = 0;
    pad_N = 0;
    if mod(M, num_tiles(1)) ~= 0
        pad_M = num_tiles(1) - mod(M, num_tiles(1));
    end
    if mod(N, num_tiles(2)) ~= 0
        pad_N = num_tiles(2) - mod(N, num_tiles(2));
    end
    p_image = padarray(image, [pad_M, pad_N], 'replicate', 'post');
end

function [M_block, N_block] = calculate_block_size(p_image, num_tiles)
    % 计算每个块的大小
    [M, N] = size(p_image);
    M_block = M / num_tiles(1);
    N_block = N / num_tiles(2);
end

function [xc, yc] = center_of_block(k, m, block_size)
    % 计算块的中心像素坐标
    M_block = block_size(1);
    N_block = block_size(2);
    xc = (k - 1) * M_block + (M_block + 1) / 2;
    yc = (m - 1) * N_block + (N_block + 1) / 2;
end

function [k, m] = block_index(i, j, block_size)
    % 计算像素所在的块索引
    M_block = block_size(1);
    N_block = block_size(2);
    k = ceil(i / M_block);
    m = ceil(j / N_block);
end

function [cdf_matrix] = make_cdf_matrix(p_image, block_size, limit)
    % 为每个块创建CDF矩阵
    [M, N] = size(p_image);
    M_block = block_size(1);
    N_block = block_size(2);
    T1 = M / M_block;
    T2 = N / N_block;
    max_gray_value = max(p_image(:));
    min_gray_value = min(p_image(:));
    cdf_matrix = zeros(T1, T2, 65534);

    for i = 1:T1
        for j = 1:T2
            % 提取当前块
            a = (i - 1) * M_block + 1;
            b = i * M_block;
            c = (j - 1) * N_block + 1;
            d = j * N_block;
            current_block = uint16(p_image(a:b, c:d));

            % 计算直方图
            frequency = calculate_frequency(current_block);
            [min_value, max_value, gray_level] = find_histogram_range(frequency);
            fprintf("gray_min = %d ,gary_max = %d ,gray_level = %d ,gray_grade = %d\n", min_value, max_value, gray_level, max_value - min_value + 1);

            % 裁剪直方图
            limit = floor(limit * numel(current_block));
            frequency = clip_histogram(frequency, limit, min_value, max_value);

            % 归一化直方图
            sum_val = sum(frequency);
            frequency = double(frequency) ./ double(sum_val);

            % 计算CDF
            cdf = zeros(1, 65534);
            cdf(1, 1) = frequency(1, 1);
            for m = 2:65534
                cdf(1, m) = cdf(1, m - 1) + frequency(1, m);
            end

            % 调整CDF
            if gray_level < 255
                cdf2 = cdf * gray_level;
            else
                cdf2 = cdf * 255;
            end

            cdf_matrix(i, j, :) = cdf2;
        end
    end
end

function frequency = calculate_frequency(block)
    % 计算块的灰度频率
    [row, col] = size(block);
    frequency = zeros(1, 65534);
    for m = 1:row
        for n = 1:col
            index = block(m, n);
            frequency(1, index) = frequency(1, index) + 1;
        end
    end
end

function [min_value, max_value, gray_level] = find_histogram_range(frequency)
    % 找到直方图的有效范围和灰度级别
    min_value = -1;
    max_value = -1;
    gray_level = 0;
    for m = 1:65534
        if frequency(m) ~= 0
            if min_value == -1
                min_value = m;
            end
            gray_level = gray_level + 1;
            max_value = m;
        end
    end
end

function frequency = clip_histogram(frequency, limit, min_value, max_value)
    % 裁剪直方图
    sum_val = 0;
    for m = min_value:max_value
        if frequency(m) > limit
            sum_val = sum_val + (frequency(m) - limit);
            frequency(m) = limit;
        end
    end

    % 重新分配裁剪后的像素
    while sum_val > 0
        for m = min_value:max_value
            if frequency(m) < limit
                frequency(m) = frequency(m) + 1;
            end
            sum_val = sum_val - 1;
            if sum_val <= 0
                break;
            end
        end
    end

end

function is_frame = is_in_frame(i, j, xc11, yc11, xcT1T2, ycT1T2)
    % 检查像素是否在边框内
    is_frame = (i < xc11 || i >= xcT1T2 || j < yc11 || j >= ycT1T2);
end

function is_corner = is_corner_subblock(i, j, xc11, yc11, xcT1T2, ycT1T2)
    % 检查像素是否在角块内
    is_corner = (i < xc11 && j < yc11) || (i >= xcT1T2 && j < yc11) || (i < xc11 && j >= ycT1T2) || (i >= xcT1T2 && j >= ycT1T2);
end

function value = linear_interpolation(i, j, k, m, p_image, cdf_matrix, block_size, num_tiles ,yc11,ycT1T2)
    % 线性插值
    M_block = block_size(1);
    N_block = block_size(2);
    [xc, yc] = center_of_block(k, m, block_size);
    if (k == 1 || k == num_tiles(1)) && (j >= yc11 && j < ycT1T2)
        if j < yc
            a = j - (yc - N_block);
            b = yc - j;
            value = (a * cdf_matrix(k, m, p_image(i, j)) + b * cdf_matrix(k, m - 1, p_image(i, j))) / (a + b);
        else
            a = j - yc;
            b = (yc + N_block) - j;
            value = (b * cdf_matrix(k, m, p_image(i, j)) + a * cdf_matrix(k, m + 1, p_image(i, j))) / (a + b);
        end
    else
        if i < xc
            a = xc - i;
            b = i - (xc - M_block);
            value = (a * cdf_matrix(k - 1, m, p_image(i, j)) + b * cdf_matrix(k, m, p_image(i, j))) / (a + b);
        else
            a = (xc + M_block) - i;
            b = i - xc;
            value = (a * cdf_matrix(k, m, p_image(i, j)) + b * cdf_matrix(k + 1, m, p_image(i, j))) / (a + b);
        end
    end
end

function value = bilinear_interpolation(i, j, k, m, p_image, cdf_matrix, block_size)
    % 双线性插值
    M_block = block_size(1);
    N_block = block_size(2);
    [xc, yc] = center_of_block(k, m, block_size);
    if i < xc && j < yc
        a = j - (yc - N_block);
        b = yc - j;
        c = i - (xc - M_block);
        d = xc - i;
        sh1 = (b * cdf_matrix(k - 1, m - 1, p_image(i, j)) + a * cdf_matrix(k - 1, m, p_image(i, j))) / (a + b);
        sh2 = (b * cdf_matrix(k, m - 1, p_image(i, j)) + a * cdf_matrix(k, m, p_image(i, j))) / (a + b);
        value = (d * sh1 + c * sh2) / (c + d);
    elseif i >= xc && j < yc
        a = j - (yc - N_block);
        b = yc - j;
        c = i - xc;
        d = xc + M_block - i;
        sh1 = (b * cdf_matrix(k, m - 1, p_image(i, j)) + a * cdf_matrix(k, m, p_image(i, j))) / (a + b);
        sh2 = (b * cdf_matrix(k + 1, m - 1, p_image(i, j)) + a * cdf_matrix(k + 1, m, p_image(i, j))) / (a + b);
        value = (d * sh1 + c * sh2) / (c + d);
    elseif i < xc && j >= yc
        a = j - yc;
        b = yc + N_block - j;
        c = i - (xc - M_block);
        d = xc - i;
        sh1 = (b * cdf_matrix(k - 1, m, p_image(i, j)) + a * cdf_matrix(k - 1, m + 1, p_image(i, j))) / (a + b);
        sh2 = (b * cdf_matrix(k, m, p_image(i, j)) + a * cdf_matrix(k, m + 1, p_image(i, j))) / (a + b);
        value = (d * sh1 + c * sh2) / (c + d);
    else
        a = j - yc;
        b = yc + N_block - j;
        c = i - xc;
        d = xc + M_block - i;
        sh1 = (b * cdf_matrix(k, m, p_image(i, j)) + a * cdf_matrix(k, m + 1, p_image(i, j))) / (a + b);
        sh2 = (b * cdf_matrix(k + 1, m, p_image(i, j)) + a * cdf_matrix(k + 1, m + 1, p_image(i, j))) / (a + b);
        value = (d * sh1 + c * sh2) / (c + d);
    end
end

function save_image(image, save_dir, suffix)
    save_temp_dir = strcat(save_dir ,suffix );
    save_temp_dir = char(save_temp_dir(1));

    % 保存图像到文件
    fileID = fopen(save_temp_dir, 'wb');
    fwrite(fileID, uint16(image), 'uint16');
    fclose(fileID);
end
    