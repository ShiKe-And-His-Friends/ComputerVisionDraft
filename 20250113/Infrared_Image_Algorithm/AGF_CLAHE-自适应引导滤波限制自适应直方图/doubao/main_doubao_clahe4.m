clc;
clear;

cols = 640;
rows = 512;

%��ȡ14bits��doubleС��rawͼ
%fid = fopen( "C:\MATLAB_CODE\input_image\����3.raw", 'r');
fid = fopen( "C:\Users\shike\Desktop\03����\03\x1.raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
I = reshape(rawData,cols ,rows);
I = I - 16383;

% ���� ��/��?
mu = mean(I(:));
sigma_squared = var(I(:));
climp_limit = sqrt(sigma_squared)/ mu ;
climp_limit = mu / sqrt(sigma_squared);
climp_limit = 0.048;
fprintf("������ֵ %f\n",climp_limit);

% ����ģʽ��ʵʱԤ����
O_fast = CLAHE_16to8(I, 32, climp_limit, true); % ��16��16��������ǿ
fileID = fopen('C:\Users\shike\Desktop\clahe_fast.raw', 'wb');
fwrite(fileID,O_fast , 'uint8'); 
fclose(fileID);

% ��ȷģʽ�����������
O_precise = CLAHE_16to8(I, 32, climp_limit, false); % ��32��32��������
fileID = fopen('C:\Users\shike\Desktop\clahe_precise.raw', 'wb');
fwrite(fileID,O_precise , 'uint8'); 
fclose(fileID);

% ��ʾ�Ա�
figure;
subplot(1,3,1); imshow(I, []); title('ԭʼ16λ����');
subplot(1,3,2); imshow(O_fast); title('����ģʽ��8λ��');
subplot(1,3,3); imshow(O_precise); title('��ȷģʽ��8λ��');


function O = CLAHE_16to8(I, block_size, climp_limit, fast_mode)
    % I: 16λ����ͼ��uint16��0-65535��
    % block_size: ���С��Ĭ��16��
    % clip_limit: �ü���ֵ��ռ�����ر�����Ĭ��0.01��
    % fast_mode: �Ƿ����ģʽ��Ĭ��true��

    pad = 0;
    
    %% 1. Ԥ����16λ��256bins + �Գ����
    I_bin = uint16(I); 
    I_pad = padarray(I_bin, [pad pad], 'symmetric', 'both'); % �Գ����
    sorted_block_I = I;
    
    sort_blcok =gray_level_sort(I_pad);
    I_pad = sort_blcok;
    
    %% 2. �ֿ����ӳ���16λ��8λ��
    [H, W] = size(I_pad);
    map = cell(H/block_size, W/block_size); % �洢ÿ�����ӳ���65536���ȣ�

    for i = 1:block_size:H 
        for j = 1:block_size:W 
                
            block = I_pad(i:i+block_size-1, j:j+block_size-1);
%             sort_blcok =gray_level_part_sort(block);
%             block = sort_blcok;
            sorted_block_I(i:i+block_size-1, j:j+block_size-1) = block;
            
            % ���� ��/��?
            block_d = double(block);
            mu = mean(block_d(:));
            sigma_squared = var(block_d(:));
            sigma_all_squared = mean(var(double(I_pad)));
            climp_limit_value = mu / sigma_squared;
            
            hist = calc_hist(block, 16384); % ����256binsֱ��ͼ
            [hist,gray_level] = clip_hist(hist, climp_limit, block_size); % �ü�
            cdf = normalize_cdf(hist); % ����16��8ӳ���
            if gray_level < 255
                cdf = cdf * gray_level;
            else
                cdf = cdf * 255;
            end
            
            map{(i-1)/block_size+1, (j-1)/block_size+1} = cdf;
        end
    end

    %% 3. ��̬��Χѹ��+��ֵ
    if fast_mode
        O = fast_interpolate(sorted_block_I, map, block_size, pad);
    else
        O = precise_interpolate(sorted_block_I, map, block_size, pad);
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

function O = precise_interpolate(I, map, blockSize, overlap)
    [H, W] = size(I);
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
            [neighborY, neighborX] = getNeighbors(yIndex, xIndex, numBlocks, size(map, 1), size(map, 2));
            
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
                mappedValue(i) = map{uint16(neighborY(i)), uint16(neighborX(i))}(originalValue);
            end

            O(y,x) = uint8(sum(weights .* mappedValue));
        end
    end
    
end 

function O = fast_interpolate(I, map, block, pad)
    [H, W] = size(I);
    O = zeros(H, W, 'uint8');
    fid = fopen("C:\Users\shike\Desktop\a.txt","w");
    for y = 1:H
        for x = 1:W
            % ��λ�����������㣩
            i = (y + pad) / block; % �����Ŀ������꣨MATLAB��1��ʼ��
            j = (x + pad) / block;
            i1 = ceil(i)-1; i2 = ceil(i)+1; j1 = ceil(j)-1; j2 = ceil(j)+1;
            
            % �߽紦��
            i1 = max(1, min(i1, size(map,1)));
            i2 = max(1, min(i2, size(map,1)));
            j1 = max(1, min(j1, size(map,2)));
            j2 = max(1, min(j2, size(map,2))); % �ұ߽粻����

            % ��ȡ4�����ӳ��ֵ
            val16 = I(y,x);
            index_vall6 = uint16(val16 + 1);
            v11 = map{i1,j1}(index_vall6); % 16λ����+1��MATLAB��1��ʼ��
            v12 = map{i1,j2}(index_vall6);
            v21 = map{i2,j1}(index_vall6);
            v22 = map{i2,j2}(index_vall6);
            
%             if v11 == 0
%                 v11 = min(map{i1,j1}(map{i1,j1}~=0));
%             end
%             if v12 == 0
%                 v12 = min(map{i1,j2}(map{i1,j2}~=0));
%             end
%             if v21 == 0
%                 v21 = min(map{i2,j1}(map{i2,j1}~=0));
%             end
%             if v22 == 0
%                 v22 = min(map{i2,j2}(map{i2,j2}~=0));
%             end
            
            [v11_m ,v11_n ]= block_center(i1,j1,block);
            [v12_m ,v12_n ]= block_center(i1,j2,block);
            [v21_m ,v21_n ]= block_center(i2,j1,block);
            [v22_m ,v22_n]= block_center(i2,j2,block);
            v11_m = v11_m - block/2;v11_n = v11_n - block/2;
            v12_m = v12_m - block/2;v12_n = v12_n + block/2;
            v21_m = v21_m + block/2;v21_n = v21_n - block/2;
            v22_m = v22_m + block/2;v22_n = v22_n + block/2;
            
%             if (v11_m > y || y > v22_m) && i1~= 1 && i2~= 20
%                 fprintf(fid,"block center %d %d  %d %d\n",i1,j1,i2,j2);
%                 fprintf(fid ,"�� y%d  x%d  %f %f | %f %f\n\n",y ,x,v11_m ,v21_m,v12_m,v22_m);
%             end
%             
%              if (v11_n > x || x > v22_n) && j1~= 1 && j2~= 16
%                  fprintf(fid,"block center %d %d  %d %d\n",i1,j1,i2,j2);
%                  fprintf(fid ,"�� y%d x%d  %f %f | %f %f\n\n",y ,x,v11_n,v12_n,v21_n,v22_n);
%              end
            
            % ˫���Բ�ֵȨ��
            dx1 = (v22_m - y) / (v22_m - v11_m);
            dx2 = (y - v11_m) / (v22_m - v11_m);
            dy1 = (v22_n - x) / (v22_n - v11_n);
            dy2 = (x - v11_n) / (v22_n - v11_n);
            
            dd11 = sqrt((y-v11_m)*(y-v11_m) + (x-v11_n)*(x-v11_n));
            dd12 = sqrt((y-v12_m)*(y-v12_m) + (x-v12_n)*(x-v12_n));
            dd21 = sqrt((y-v21_m)*(y-v21_m) + (x-v21_n)*(x-v21_n));
            dd22 = sqrt((y-v22_m)*(y-v22_m) + (x-v22_n)*(x-v22_n));
            
            numbers = [dd11 ,dd12 ,dd21 ,dd22];
            sorted_ascending = sort(numbers);
            sorted_descending = sort(sorted_ascending, 'descend');
            sorted_descending = sorted_descending / sum(sorted_descending);
            w11 = sorted_descending(1);
            w12 = sorted_descending(2);
            w21 = sorted_descending(3);
            w22 = sorted_descending(4);

%             fprintf(fid ,"x%d y%d\n" ,x ,y);
%             fprintf(fid ,"mapij %d %d %d %d \n" ,i1,j1,i2,j2);
             fprintf(fid ,"dxdy %f %f %f %f \n" ,dx1,dx2,dy1 ,dy2);
            weight = [
                
            dx1 * dy1  ,   dx2 *dy1  , dx1 * dy2 ,  dx2 * dy2
            
            ];
 
            % ת��Ϊ�������ͽ��м���
            weight_double = double(weight);
            v_double = double([v11; v12; v21; v22]);

            % ˫���Բ�ֵ
            O(y,x) = uint8(round(weight_double * v_double));
            
             % ����ֿ�����
            y1 = floor((y - 1) / block) + 1;
            y2 = min(y1 + 1, size(map, 1));
            x1 = floor((x - 1) / block) + 1;
            x2 = min(x1 + 1, size(map, 2));
            
            v11 = map{y1,x1}(index_vall6); % 16λ����+1��MATLAB��1��ʼ��
            v12 = map{y1,x2}(index_vall6);
            v21 = map{y2,x1}(index_vall6);
            v22 = map{y2,x2}(index_vall6);
            
            % �����ֵȨ��
            wy = (y - (y1 - 1) * block) / block;
            wx = (x - (x1 - 1) * block) / block;
            
            mapY1 = (1 - wy) * v11 + wy * v21;
            mapY2 = (1 - wy) * v12 + wy * v22;
            mappedValue = (1 - wx) * mapY1 + wx * mapY2;
            O(y,x) = uint8(round(mappedValue));
%             if i1 == 10 && j1 == 2
%                 if y == 340
%                 if x == 88
%                     O(y,x) = uint8(255);
%                     fprintf(fid ,"input %d\n" ,val16);
%                     fprintf(fid ,"x%d y%d\n" ,x ,y);
%                     fprintf(fid ,"mapij %d %d %d %d \n" ,i1,j1,i2,j2);
%                     fprintf(fid ,"weight %f %f %f %f \n" ,weight(1),weight(2),weight(3),weight(4));
%                     fprintf(fid ,"value %f %f %f %f \n" ,v_double(1),v_double(2),v_double(3),v_double(4));
%                     fprintf(fid ,"weight_i*value_i %f %f %f %f \n" ,weight(1)*v_double(1),weight(2)*v_double(2),weight(3)*v_double(3),weight(4)*v_double(4));
%                     fprintf(fid ,"out %f\n\n" ,weight_double * v_double);
%                     map_infact_v11 = map{i1,j1};
%                     map_infact_v12 = map{i1,j2};
%                     map_infact_v21 = map{i2,j1};
%                     map_infact_v22 = map{i2,j2};
%                     
%                     fprintf("block center %d %d  %d %d\n",i1,j1,i2,j2);
%                     fprintf("y%d  %f %f | %f %f\n",y,v11_m ,v21_m,v12_m,v22_m);
%                     fprintf("x%d  %f %f | %f %f\n\n",x,v11_n,v12_n,v21_n,v22_n);
%                     fprintf("%f\n" ,weight(4));
%                     
%                 end
%             end

        end
    end
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
%         gray_level_sorted_image = newArray - min_value + 1;

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

function gray_level = search_gray_level(vv)
    find_nums = 0;
    for i = 1 : length(vv)
        if vv(1,i) ~= 0
            find_nums = find_nums + 1;
        end
    end
    gray_level = find_nums;
end

function [m1,n1] = block_center(i ,j ,block)
    m1 = (i-0.5) * block;
    n1 = (j-0.5) * block;
end

% ���½ǵ�4����
% function O = fast_interpolate(I, map, block, pad)
%     [H, W] = size(I);
%     O = zeros(H, W, 'uint8');
% 
%     for y = 1:uint16(H)
%         for x = 1:uint16(W)
%             % ��λ�����������㣩
%             i = double(y + pad) / block; % �����Ŀ������꣨MATLAB��1��ʼ��
%             j = double(x + pad) / block;
%             i1 = ceil(i); i2 = i1+1; j1 = ceil(j); j2 = j1+1;
% 
%             % �߽紦��
%             i1 = max(1, min(i1, size(map,1)));
%             i2 = max(1, min(i2, size(map,1)));
%             j1 = max(1, min(j1, size(map,2)));
%             j2 = max(1, min(j2, size(map,2))); % �ұ߽粻����
%            
%             % ��ȡ4�����ӳ��ֵ
%             val16 = I(y,x);
%             v11 = map{i1,j1}(val16); % 16λ����+1��MATLAB��1��ʼ��
%             v12 = map{i1,j2}(val16);
%             v21 = map{i2,j1}(val16);
%             v22 = map{i2,j2}(val16);
% 
%             % ˫���Բ�ֵȨ��
%             dx = double(y) - double(i1) * block + block; dy = double(x) - double(j1) * block + block;
%            
%             dx = (block-dx+1)/(block+1);
%             dy = (block-dy+1)/(block+1);
%             
%             %dx*dy,     dx*(1 - dy),     (1 - dx)*dy,     (1 - dx)*(1 - dy)
%             weight = [dx*dy,     dx*(1 - dy),     (1 - dx)*dy,     (1 - dx)*(1 - dy)];
%             
%             % ת��Ϊ�������ͽ��м���
%             weight_double = double(weight);
%             v_double = double([v11; v12; v21; v22]);
% 
%             % ˫���Բ�ֵ
%             O(y,x) = uint8(round(weight_double * v_double));
%         end
%     end
% 
% end

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
