clc;
clear;

cols = 640;
rows = 512;

%读取14bits的double小端raw图
%fid = fopen( "C:\MATLAB_CODE\input_image\场景5.raw", 'r');
fid = fopen( "C:\Users\shike\Desktop\03数据\03\x1.raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
I = reshape(rawData,cols ,rows);
I = I - 16383;
%I = floor(uint16(I)/4+1);
% 计算 μ/σ?
mu = mean(I(:));
sigma_squared = var(I(:));
climp_limit = 5 * mu / sqrt(sigma_squared);
climp_limit = mu / sigma_squared;
climp_limit = 0.048;
fprintf("抑制阈值 %f\n",climp_limit);

% 快速模式（实时预览）
O_fast = CLAHE_16to8(I, 64, 0.0015, true); % 块16×16，低温增强
fileID = fopen('C:\Users\shike\Desktop\clahe_fast.raw', 'wb');
fwrite(fileID,O_fast , 'uint8'); 
fclose(fileID);

% 精确模式（最终输出）
O_precise = CLAHE_16to8(I, 64, 0.0015, false); % 块32×32，宽温区
fileID = fopen('C:\Users\shike\Desktop\clahe_precise.raw', 'wb');
fwrite(fileID,O_precise , 'uint8'); 
fclose(fileID);

% 显示对比
figure;
subplot(1,3,1); imshow(I, []); title('原始16位红外');
subplot(1,3,2); imshow(O_fast); title('快速模式（8位）');
subplot(1,3,3); imshow(O_precise); title('精确模式（8位）');


function O = CLAHE_16to8(infraredImage, blockSize, clipLimit, fast_mode)
    % 输入图像的尺寸
    [height, width] = size(infraredImage);
    
    % 重叠分块
    %overlap = blockSize / 4;
    overlap = blockSize / 4;
    overlap = 0;
    paddedImage = padarray(infraredImage, [overlap, overlap], 'symmetric');
    paddedHeight = height + 2 * overlap;
    paddedWidth = width + 2 * overlap;
    
    sort_blcok =gray_level_sort(paddedImage);
    paddedImage = sort_blcok;
    
    % 初始化映射表
    map = zeros(2^14, ceil(paddedHeight / blockSize), ceil(paddedWidth / blockSize));
    
    % 遍历每个分块
    for y = 1:blockSize:paddedHeight - blockSize + 1
        for x = 1:blockSize:paddedWidth - blockSize + 1
            % 提取分块
            block = paddedImage(y:y+blockSize-1, x:x+blockSize-1);
            
            % 自适应剪裁阈值调整
            block_d = double(block);
            mu = mean(block_d(:));
            sigma_squared = var(block_d(:));
            sigma_all_squared = mean(var(double(infraredImage)));
            %localClipLimit = clipLimit * (localVariance / mean(var(infraredImage(:))));
            localClipLimit = mu / sigma_squared;
            
            localVariance = var(block(:));
            localClipLimit = clipLimit * (localVariance / sigma_all_squared);
            
            hist = calc_hist(block, 16384); % 计算256bins直方图
            [hist,gray_level] = clip_hist(hist, localClipLimit, blockSize); % 裁剪
            cdf = normalize_cdf(hist); % 生成16→8映射表
            if gray_level < 255
                cdf = cdf * gray_level;
            else
                cdf = cdf * 255;
            end
                  
            % 生成映射表
            map(:, (y - 1) / blockSize + 1, (x - 1) / blockSize + 1) = cdf;
        end
    end
    
     %% 3. 动态范围压缩+插值
    if fast_mode
        O = fast_interpolate(paddedImage, map, blockSize, overlap);
    else
        O = precise_interpolate(paddedImage, map, blockSize, overlap);
    end
    
    O = uint8(max(min(O, 255), 0)); % 截断到0-255
    
end

function hist = calc_hist(arrys, bins)
    hist_info = zeros(1,bins);
    [H, W] = size(arrys);
    for i = 1:H
        for j = 1:W
            val = arrys(i,j);
            hist_info(1,val) = hist_info(1,val) + 1;
        end
    end
    %直方图按照个数或者按照频率
    %hist = double(hist_info) /(H*W);
    hist = hist_info;
end

function [hist_limit,gray_level] = clip_hist(hist, clip, block_size)
    %找到直方图均衡的区间
    min_value = -1;
    max_value = -1;
    for m = 1:length(hist)
        if hist(m)<=0
            continue;
        end
        min_value = m;
        break;
    end
    m = length(hist);
    while(max_value==-1)
        if hist(m)<=0
            m = m -1;
            continue;
        end
        max_value = m;
    end     
    gray_level = 0;
    for m = 1:length(hist)
        if hist(m)<=0
            continue;
        end
        gray_level = gray_level+1;
    end
    
    % clip the normalized histogram 
    sum_val = 0;
    limit =  clip * block_size^2;
    for m = min_value : max_value
        if (hist(m)>limit) 
            sum_val = sum_val + (hist(m)-limit);
            hist(m) = limit;
        end
    end

    %除最重复值外平均分配
    while(sum_val > 0)
        for m = min_value : max_value
            if hist(m) == 0
                continue;
            end
            if hist(m) < limit
                hist(m) = hist(m) + 1;
            end
            sum_val = sum_val - 1;
            if (sum_val <= 0)
                break;
            end
        end
    end
    hist_limit = hist;
end

function cdf = normalize_cdf(hist)
    % 生成65536长度的映射表（每个bin对应256个16位值）
    cdf = cumsum(hist);
    cdf = (cdf - cdf(1)) / (cdf(end) - cdf(1));
end

function gray_level_sorted_image = gray_level_sort(input_image)
%         imageArray = input_image;
%         min_value = min(min(imageArray(:)));
%         average_value = mean(mean(imageArray(:)));
%         % 计算灰度级别
%         grayLevels = unique(imageArray);
%         grayLevels = sort(grayLevels);
% 
%         % 按照灰度级别从1递增进行处理
%         newArray = zeros(size(imageArray), 'uint16');
%         for i = 1:length(grayLevels)
%             level = grayLevels(i);
%             newArray(imageArray == level) = i;
%         end
% 
%         gray_level_sorted_image = double(newArray - min_value + 1);

        imageArray = input_image;
        min_value = min(min(imageArray(:)));
        average_value = mean(mean(imageArray(:)));
        gray_level_sorted_image = imageArray - min_value + 1;

%     % 显示原始数组和处理后的数组（以图像形式展示更直观）
%     figure;
%     subplot(1,2,1);
%     imshow(imageArray, [0, 16383]);
%     title('原始数组');
%     subplot(1,2,2);
%     imshow(newArray, [1, length(grayLevels)]);
%     title('按照灰度级别从1递增处理后的数组');
end

function [m1,n1] = block_center(i ,j ,block)
    m1 = (i-0.5) * block;
    n1 = (j-0.5) * block;
end

function O = fast_interpolate(paddedImage, map, blockSize, overlap)
    
    [height, width] = size(paddedImage);

    height = height - 2*overlap;
    width = width - 2*overlap;
    
    % 高阶插值（双三次插值）
    enhancedImage = zeros(height, width, 'uint8');
    for y = 1:height
        for x = 1:width
            % 定位像素在填充图像中的位置
            py = y + overlap;
            px = x + overlap;
            
            % 计算分块索引
            y1 = floor((py - 1) / blockSize) + 1;
            y2 = min(y1 + 1, size(map, 2));
            x1 = floor((px - 1) / blockSize) + 1;
            x2 = min(x1 + 1, size(map, 3));
            
            % 计算插值权重
            wy = (py - (y1 - 1) * blockSize) / blockSize;
            wx = (px - (x1 - 1) * blockSize) / blockSize;
            
            % 双三次插值
            map11 = map(:, y1, x1);
            map12 = map(:, y1, x2);
            map21 = map(:, y2, x1);
            map22 = map(:, y2, x2);
            
            mapY1 = (1 - wy) * map11 + wy * map21;
            mapY2 = (1 - wy) * map12 + wy * map22;
            mappedValue = (1 - wx) * mapY1 + wx * mapY2;
            
            % 获取映射后的值
            originalValue = paddedImage(py, px);
            enhancedImage(y, x) = mappedValue(originalValue + 1);
        end
    end
    
    O = enhancedImage;
    
end

function O = precise_interpolate(I, map, blockSize, overlap)
    [H, W] = size(I);
    
    H = H - 2*overlap;
    W = W - 2*overlap;
    
    O = zeros(H, W, 'uint8');
    for y = 1:H
        for x = 1:W
            % 定位像素在填充图像中的位置
            py = y + overlap;
            px = x + overlap;
            
            % 计算分块索引
            yIndex = ceil(py / blockSize);
            xIndex = ceil(px / blockSize);
            
            % 获取周围子块的中心坐标
            numBlocks = 9;
            [neighborY, neighborX] = getNeighbors(yIndex, xIndex, numBlocks, size(map, 2), size(map, 3));
            
            % 计算高斯权重
            weights = zeros(size(neighborY));
            for i = 1:length(neighborY)
                [v11_m ,v11_n ]= block_center(neighborY(i),neighborX(i),blockSize);
                dist = sqrt((py - v11_m)^2 + (px - v11_n)^2);
                weights(i) = exp(-dist^2 / (2 * (blockSize / 2)^2));
            end
            weights = weights / sum(weights);
            
            % 加权求和
            originalValue = I(py, px);
            mappedValue = zeros(1, length(neighborY));
            for i = 1:length(neighborY)
                mappedValue(i) = map( originalValue , uint16(neighborY(i)), uint16(neighborX(i)) );
            end

            O(y,x) = uint8(sum(weights .* mappedValue));
        end
    end
    
end 


function [neighborY, neighborX] = getNeighbors(yIndex, xIndex, numBlocks, maxY, maxX)
    % 获取周围子块的中心坐标
    neighborY = [];
    neighborX = [];
    for dy = -1:1
        for dx = -1:1
            ny = yIndex + dy;
            nx = xIndex + dx;
            if ny >= 1 && ny <= maxY && nx >= 1 && nx <= maxX
                neighborY = [neighborY, ny];
                neighborX = [neighborX, nx];
            end
            if length(neighborY) >= numBlocks
                break;
            end
        end
        if length(neighborY) >= numBlocks
            break;
        end
    end
end
