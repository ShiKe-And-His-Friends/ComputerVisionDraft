clc;
clear;

cols = 640;
rows = 512;

%��ȡ14bits��doubleС��rawͼ
%fid = fopen( "C:\MATLAB_CODE\input_image\����5.raw", 'r');
fid = fopen( "C:\Users\shike\Desktop\03����\03\x1.raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
I = reshape(rawData,cols ,rows);
I = I - 16383;
%I = floor(uint16(I)/4+1);
% ���� ��/��?
mu = mean(I(:));
sigma_squared = var(I(:));
climp_limit = 5 * mu / sqrt(sigma_squared);
climp_limit = mu / sigma_squared;
climp_limit = 0.048;
fprintf("������ֵ %f\n",climp_limit);

% ����ģʽ��ʵʱԤ����
O_fast = CLAHE_16to8(I, 64, 0.0015, true); % ��16��16��������ǿ
fileID = fopen('C:\Users\shike\Desktop\clahe_fast.raw', 'wb');
fwrite(fileID,O_fast , 'uint8'); 
fclose(fileID);

% ��ȷģʽ�����������
O_precise = CLAHE_16to8(I, 64, 0.0015, false); % ��32��32��������
fileID = fopen('C:\Users\shike\Desktop\clahe_precise.raw', 'wb');
fwrite(fileID,O_precise , 'uint8'); 
fclose(fileID);

% ��ʾ�Ա�
figure;
subplot(1,3,1); imshow(I, []); title('ԭʼ16λ����');
subplot(1,3,2); imshow(O_fast); title('����ģʽ��8λ��');
subplot(1,3,3); imshow(O_precise); title('��ȷģʽ��8λ��');


function O = CLAHE_16to8(infraredImage, blockSize, clipLimit, fast_mode)
    % ����ͼ��ĳߴ�
    [height, width] = size(infraredImage);
    
    % �ص��ֿ�
    %overlap = blockSize / 4;
    overlap = blockSize / 4;
    overlap = 0;
    paddedImage = padarray(infraredImage, [overlap, overlap], 'symmetric');
    paddedHeight = height + 2 * overlap;
    paddedWidth = width + 2 * overlap;
    
    sort_blcok =gray_level_sort(paddedImage);
    paddedImage = sort_blcok;
    
    % ��ʼ��ӳ���
    map = zeros(2^14, ceil(paddedHeight / blockSize), ceil(paddedWidth / blockSize));
    
    % ����ÿ���ֿ�
    for y = 1:blockSize:paddedHeight - blockSize + 1
        for x = 1:blockSize:paddedWidth - blockSize + 1
            % ��ȡ�ֿ�
            block = paddedImage(y:y+blockSize-1, x:x+blockSize-1);
            
            % ����Ӧ������ֵ����
            block_d = double(block);
            mu = mean(block_d(:));
            sigma_squared = var(block_d(:));
            sigma_all_squared = mean(var(double(infraredImage)));
            %localClipLimit = clipLimit * (localVariance / mean(var(infraredImage(:))));
            localClipLimit = mu / sigma_squared;
            
            localVariance = var(block(:));
            localClipLimit = clipLimit * (localVariance / sigma_all_squared);
            
            hist = calc_hist(block, 16384); % ����256binsֱ��ͼ
            [hist,gray_level] = clip_hist(hist, localClipLimit, blockSize); % �ü�
            cdf = normalize_cdf(hist); % ����16��8ӳ���
            if gray_level < 255
                cdf = cdf * gray_level;
            else
                cdf = cdf * 255;
            end
                  
            % ����ӳ���
            map(:, (y - 1) / blockSize + 1, (x - 1) / blockSize + 1) = cdf;
        end
    end
    
     %% 3. ��̬��Χѹ��+��ֵ
    if fast_mode
        O = fast_interpolate(paddedImage, map, blockSize, overlap);
    else
        O = precise_interpolate(paddedImage, map, blockSize, overlap);
    end
    
    O = uint8(max(min(O, 255), 0)); % �ضϵ�0-255
    
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
    %ֱ��ͼ���ո������߰���Ƶ��
    %hist = double(hist_info) /(H*W);
    hist = hist_info;
end

function [hist_limit,gray_level] = clip_hist(hist, clip, block_size)
    %�ҵ�ֱ��ͼ���������
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

    %�����ظ�ֵ��ƽ������
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
    % ����65536���ȵ�ӳ���ÿ��bin��Ӧ256��16λֵ��
    cdf = cumsum(hist);
    cdf = (cdf - cdf(1)) / (cdf(end) - cdf(1));
end

function gray_level_sorted_image = gray_level_sort(input_image)
%         imageArray = input_image;
%         min_value = min(min(imageArray(:)));
%         average_value = mean(mean(imageArray(:)));
%         % ����Ҷȼ���
%         grayLevels = unique(imageArray);
%         grayLevels = sort(grayLevels);
% 
%         % ���ջҶȼ����1�������д���
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

%     % ��ʾԭʼ����ʹ��������飨��ͼ����ʽչʾ��ֱ�ۣ�
%     figure;
%     subplot(1,2,1);
%     imshow(imageArray, [0, 16383]);
%     title('ԭʼ����');
%     subplot(1,2,2);
%     imshow(newArray, [1, length(grayLevels)]);
%     title('���ջҶȼ����1��������������');
end

function [m1,n1] = block_center(i ,j ,block)
    m1 = (i-0.5) * block;
    n1 = (j-0.5) * block;
end

function O = fast_interpolate(paddedImage, map, blockSize, overlap)
    
    [height, width] = size(paddedImage);

    height = height - 2*overlap;
    width = width - 2*overlap;
    
    % �߽ײ�ֵ��˫���β�ֵ��
    enhancedImage = zeros(height, width, 'uint8');
    for y = 1:height
        for x = 1:width
            % ��λ���������ͼ���е�λ��
            py = y + overlap;
            px = x + overlap;
            
            % ����ֿ�����
            y1 = floor((py - 1) / blockSize) + 1;
            y2 = min(y1 + 1, size(map, 2));
            x1 = floor((px - 1) / blockSize) + 1;
            x2 = min(x1 + 1, size(map, 3));
            
            % �����ֵȨ��
            wy = (py - (y1 - 1) * blockSize) / blockSize;
            wx = (px - (x1 - 1) * blockSize) / blockSize;
            
            % ˫���β�ֵ
            map11 = map(:, y1, x1);
            map12 = map(:, y1, x2);
            map21 = map(:, y2, x1);
            map22 = map(:, y2, x2);
            
            mapY1 = (1 - wy) * map11 + wy * map21;
            mapY2 = (1 - wy) * map12 + wy * map22;
            mappedValue = (1 - wx) * mapY1 + wx * mapY2;
            
            % ��ȡӳ����ֵ
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
            % ��λ���������ͼ���е�λ��
            py = y + overlap;
            px = x + overlap;
            
            % ����ֿ�����
            yIndex = ceil(py / blockSize);
            xIndex = ceil(px / blockSize);
            
            % ��ȡ��Χ�ӿ����������
            numBlocks = 9;
            [neighborY, neighborX] = getNeighbors(yIndex, xIndex, numBlocks, size(map, 2), size(map, 3));
            
            % �����˹Ȩ��
            weights = zeros(size(neighborY));
            for i = 1:length(neighborY)
                [v11_m ,v11_n ]= block_center(neighborY(i),neighborX(i),blockSize);
                dist = sqrt((py - v11_m)^2 + (px - v11_n)^2);
                weights(i) = exp(-dist^2 / (2 * (blockSize / 2)^2));
            end
            weights = weights / sum(weights);
            
            % ��Ȩ���
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
    % ��ȡ��Χ�ӿ����������
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
