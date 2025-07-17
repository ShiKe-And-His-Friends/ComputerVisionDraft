clc;
clear;

cols = 640;
rows = 512;

%读取14bits的double小端raw图
fid = fopen( "C:\MATLAB_CODE\input_image\场景5.raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
I = reshape(rawData,cols ,rows);

% 快速模式（实时预览）
O_fast = CLAHE_16to8(I, 16, 0.01, true, [-20, 50]); % 块16×16，低温增强

% 精确模式（最终输出）
O_precise = CLAHE_16to8(I, 32, 0.005, false, [-40, 150]); % 块32×32，宽温区

% 显示对比
figure;
subplot(1,3,1); imshow(I, []); title('原始16位红外');
subplot(1,3,2); imshow(O_fast); title('快速模式（8位）');
subplot(1,3,3); imshow(O_precise); title('精确模式（8位）');

function O = CLAHE_16to8(I, block_size, clip_limit, fast_mode, temp_range)
% I: 16位红外图像（uint16，0-65535）
% block_size: 块大小（默认16）
% clip_limit: 裁剪阈值（占块像素比例，默认0.01）
% fast_mode: 是否快速模式（默认true）
% temp_range: [min_temp, max_temp]（默认自动计算）

if nargin < 5 || isempty(temp_range)
    temp_range = [min(I(:)), max(I(:))]; % 自动温度范围
end

%% 1. 预处理：16位→256bins + 对称填充
pad = floor(block_size/2);
max_I = max(max(I));
min_I = min(min(I));
%I_bin = uint8(255 * (double(I)-min_I) / max_I);
I_bin = uint8(I / 256); % 压缩到0-255bins（保留每个bin的256个灰度级）
I_pad = padarray(I_bin, [pad pad], 'symmetric', 'both'); % 对称填充

%% 2. 分块计算映射表（16位→8位）
[H, W] = size(I_pad);
map = cell(H/block_size, W/block_size); % 存储每个块的映射表（65536长度）

for i = 1:block_size:H
    for j = 1:block_size:W
        block = I_pad(i:i+block_size-1, j:j+block_size-1);
        hist = calc_hist(block, 256); % 计算256bins直方图
        hist = clip_hist(hist, clip_limit, block_size); % 裁剪
        cdf = normalize_cdf(hist); % 生成16→8映射表
        map{(i-1)/block_size+1, (j-1)/block_size+1} = cdf;
    end
end

%% 3. 动态范围压缩+插值
if fast_mode
    O = fast_interpolate(I, map, block_size, pad, temp_range);
else
    O = precise_interpolate(I, map, block_size, pad, temp_range);
end

O = uint8(max(min(O, 255), 0)); % 截断到0-255
end

function hist = calc_hist(block, bins)
% 输入：2D块（uint8，0-255bins）
hist = histcounts(block(:), bins, 'Normalization', 'probability');
end

function hist = clip_hist(hist, clip, block_size)
% 裁剪阈值：clip为块像素的比例（如0.01=1%）
max_cnt = clip * block_size^2;
excess = sum(hist > max_cnt);
hist = min(hist, max_cnt);
hist = hist + excess / numel(hist); % 均匀分配过剩计数
end

function cdf = normalize_cdf(hist)
% 生成65536长度的映射表（每个bin对应256个16位值）
cdf = cumsum(hist);
cdf = (cdf - cdf(1)) / (cdf(end) - cdf(1)) * 255;
cdf = repelem(uint8(cdf), 256); % 扩展为16位→8位映射
end

function O = fast_interpolate(I, map, block, pad, temp_range)
[H, W] = size(I);
O = zeros(H, W, 'uint8');
[X, Y] = meshgrid(1:W, 1:H); % 像素坐标

% 温度相关参数
T_min = temp_range(1); T_max = temp_range(2);
if T_min == T_max
    T_weight = ones(H,W);
else
    T = double(I) * (T_max - T_min) / 65535 + T_min;
    T_weight = 1 + 0.5 * tanh((T - (T_min+T_max)/2)/10); % 中心区域权重高
end

for y = 1:H
    for x = 1:W
        % 定位块索引（浮点）
        i = (y + pad - 1) / block; % 含填充的块行坐标（MATLAB从1开始）
        j = (x + pad - 1) / block;
        i1 = floor(i); i2 = ceil(i); j1 = floor(j); j2 = ceil(j);
        
        % 边界处理
        i1 = max(1, min(i1, size(map,1)));
        i2 = max(1, min(i2, size(map,1)));
        j1 = max(1, min(j1, size(map,2)));
        j2 = min(j2, size(map,2)); % 右边界不超过
        
        % 获取4个块的映射值
        val16 = I(y,x);
        v11 = map{i1,j1}(val16+1); % 16位索引+1（MATLAB从1开始）
        v12 = map{i1,j2}(val16+1);
        v21 = map{i2,j1}(val16+1);
        v22 = map{i2,j2}(val16+1);
        
        % 双线性插值权重
        dx = i - i1; dy = j - j1;
        weight = [(1-dx)*(1-dy), (1-dx)*dy, dx*(1-dy), dx*dy];
        
        % 转换为浮点类型进行计算
        weight_double = double(weight);
        v_double = double([v11; v12; v21; v22]);
        T_weight_double = double(T_weight(y,x));
        
        % 温度加权
        O(y,x) = uint8(round(weight_double * v_double * T_weight_double));
    end
end
end

function O = precise_interpolate(I, map, block, pad, temp_range)
[H, W] = size(I);
O = zeros(H, W, 'uint8');
block_num = size(map);

for y = 1:H
    for x = 1:W
        val16 = I(y,x);
        if val16 == 0 || val16 == 65535 % 处理红外盲元
            O(y,x) = median(I(max(y-1,1):min(y+1,H), max(x-1,1):min(x+1,W)));
            continue;
        end
        
        % 找到所有邻域块（最多9个）
        i_center = (y + pad - 1) / block;
        j_center = (x + pad - 1) / block;
        i_start = max(1, floor(i_center - 0.5));
        i_end = min(block_num(1), ceil(i_center + 0.5));
        j_start = max(1, floor(j_center - 0.5));
        j_end = min(block_num(2), ceil(j_center + 0.5));
        
        total_weight = 0; sum_val = 0;
        for i = i_start:i_end
            for j = j_start:j_end
                % 块中心坐标（含填充）
                block_y = (i-1)*block + pad;
                block_x = (j-1)*block + pad;
                
                % 距离权重（高斯）
                dist = sqrt((y - block_y)^2 + (x - block_x)^2);
                weight = exp(-(dist^2)/(2*(block/2)^2)); % 块内权重高
                
                % 动态范围压缩：低温区增强
                if val16 < 20000 % 假设低温区（可根据temp_range调整）
                    weight = weight * 1.5;
                end
                
                % 查表
                sum_val = sum_val + map{i,j}(val16+1) * weight;
                total_weight = total_weight + weight;
            end
        end
        
        O(y,x) = uint8(round(sum_val / total_weight));
    end
end
end
    