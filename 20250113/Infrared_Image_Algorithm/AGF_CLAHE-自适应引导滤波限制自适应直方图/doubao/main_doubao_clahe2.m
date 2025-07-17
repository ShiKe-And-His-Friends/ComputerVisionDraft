clc;
clear;

cols = 640;
rows = 512;

%��ȡ14bits��doubleС��rawͼ
fid = fopen( "C:\MATLAB_CODE\input_image\����5.raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
I = reshape(rawData,cols ,rows);

% ����ģʽ��ʵʱԤ����
O_fast = CLAHE_16to8(I, 16, 0.01, true, [-20, 50]); % ��16��16��������ǿ

% ��ȷģʽ�����������
O_precise = CLAHE_16to8(I, 32, 0.005, false, [-40, 150]); % ��32��32��������

% ��ʾ�Ա�
figure;
subplot(1,3,1); imshow(I, []); title('ԭʼ16λ����');
subplot(1,3,2); imshow(O_fast); title('����ģʽ��8λ��');
subplot(1,3,3); imshow(O_precise); title('��ȷģʽ��8λ��');

function O = CLAHE_16to8(I, block_size, clip_limit, fast_mode, temp_range)
% I: 16λ����ͼ��uint16��0-65535��
% block_size: ���С��Ĭ��16��
% clip_limit: �ü���ֵ��ռ�����ر�����Ĭ��0.01��
% fast_mode: �Ƿ����ģʽ��Ĭ��true��
% temp_range: [min_temp, max_temp]��Ĭ���Զ����㣩

if nargin < 5 || isempty(temp_range)
    temp_range = [min(I(:)), max(I(:))]; % �Զ��¶ȷ�Χ
end

%% 1. Ԥ����16λ��256bins + �Գ����
pad = floor(block_size/2);
max_I = max(max(I));
min_I = min(min(I));
%I_bin = uint8(255 * (double(I)-min_I) / max_I);
I_bin = uint8(I / 256); % ѹ����0-255bins������ÿ��bin��256���Ҷȼ���
I_pad = padarray(I_bin, [pad pad], 'symmetric', 'both'); % �Գ����

%% 2. �ֿ����ӳ���16λ��8λ��
[H, W] = size(I_pad);
map = cell(H/block_size, W/block_size); % �洢ÿ�����ӳ���65536���ȣ�

for i = 1:block_size:H
    for j = 1:block_size:W
        block = I_pad(i:i+block_size-1, j:j+block_size-1);
        hist = calc_hist(block, 256); % ����256binsֱ��ͼ
        hist = clip_hist(hist, clip_limit, block_size); % �ü�
        cdf = normalize_cdf(hist); % ����16��8ӳ���
        map{(i-1)/block_size+1, (j-1)/block_size+1} = cdf;
    end
end

%% 3. ��̬��Χѹ��+��ֵ
if fast_mode
    O = fast_interpolate(I, map, block_size, pad, temp_range);
else
    O = precise_interpolate(I, map, block_size, pad, temp_range);
end

O = uint8(max(min(O, 255), 0)); % �ضϵ�0-255
end

function hist = calc_hist(block, bins)
% ���룺2D�飨uint8��0-255bins��
hist = histcounts(block(:), bins, 'Normalization', 'probability');
end

function hist = clip_hist(hist, clip, block_size)
% �ü���ֵ��clipΪ�����صı�������0.01=1%��
max_cnt = clip * block_size^2;
excess = sum(hist > max_cnt);
hist = min(hist, max_cnt);
hist = hist + excess / numel(hist); % ���ȷ����ʣ����
end

function cdf = normalize_cdf(hist)
% ����65536���ȵ�ӳ���ÿ��bin��Ӧ256��16λֵ��
cdf = cumsum(hist);
cdf = (cdf - cdf(1)) / (cdf(end) - cdf(1)) * 255;
cdf = repelem(uint8(cdf), 256); % ��չΪ16λ��8λӳ��
end

function O = fast_interpolate(I, map, block, pad, temp_range)
[H, W] = size(I);
O = zeros(H, W, 'uint8');
[X, Y] = meshgrid(1:W, 1:H); % ��������

% �¶���ز���
T_min = temp_range(1); T_max = temp_range(2);
if T_min == T_max
    T_weight = ones(H,W);
else
    T = double(I) * (T_max - T_min) / 65535 + T_min;
    T_weight = 1 + 0.5 * tanh((T - (T_min+T_max)/2)/10); % ��������Ȩ�ظ�
end

for y = 1:H
    for x = 1:W
        % ��λ�����������㣩
        i = (y + pad - 1) / block; % �����Ŀ������꣨MATLAB��1��ʼ��
        j = (x + pad - 1) / block;
        i1 = floor(i); i2 = ceil(i); j1 = floor(j); j2 = ceil(j);
        
        % �߽紦��
        i1 = max(1, min(i1, size(map,1)));
        i2 = max(1, min(i2, size(map,1)));
        j1 = max(1, min(j1, size(map,2)));
        j2 = min(j2, size(map,2)); % �ұ߽粻����
        
        % ��ȡ4�����ӳ��ֵ
        val16 = I(y,x);
        v11 = map{i1,j1}(val16+1); % 16λ����+1��MATLAB��1��ʼ��
        v12 = map{i1,j2}(val16+1);
        v21 = map{i2,j1}(val16+1);
        v22 = map{i2,j2}(val16+1);
        
        % ˫���Բ�ֵȨ��
        dx = i - i1; dy = j - j1;
        weight = [(1-dx)*(1-dy), (1-dx)*dy, dx*(1-dy), dx*dy];
        
        % ת��Ϊ�������ͽ��м���
        weight_double = double(weight);
        v_double = double([v11; v12; v21; v22]);
        T_weight_double = double(T_weight(y,x));
        
        % �¶ȼ�Ȩ
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
        if val16 == 0 || val16 == 65535 % �������äԪ
            O(y,x) = median(I(max(y-1,1):min(y+1,H), max(x-1,1):min(x+1,W)));
            continue;
        end
        
        % �ҵ���������飨���9����
        i_center = (y + pad - 1) / block;
        j_center = (x + pad - 1) / block;
        i_start = max(1, floor(i_center - 0.5));
        i_end = min(block_num(1), ceil(i_center + 0.5));
        j_start = max(1, floor(j_center - 0.5));
        j_end = min(block_num(2), ceil(j_center + 0.5));
        
        total_weight = 0; sum_val = 0;
        for i = i_start:i_end
            for j = j_start:j_end
                % ���������꣨����䣩
                block_y = (i-1)*block + pad;
                block_x = (j-1)*block + pad;
                
                % ����Ȩ�أ���˹��
                dist = sqrt((y - block_y)^2 + (x - block_x)^2);
                weight = exp(-(dist^2)/(2*(block/2)^2)); % ����Ȩ�ظ�
                
                % ��̬��Χѹ������������ǿ
                if val16 < 20000 % ������������ɸ���temp_range������
                    weight = weight * 1.5;
                end
                
                % ���
                sum_val = sum_val + map{i,j}(val16+1) * weight;
                total_weight = total_weight + weight;
            end
        end
        
        O(y,x) = uint8(round(sum_val / total_weight));
    end
end
end
    