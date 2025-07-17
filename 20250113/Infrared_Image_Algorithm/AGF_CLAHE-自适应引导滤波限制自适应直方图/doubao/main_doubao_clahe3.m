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
O_fast = CLAHE_16to8(I, 32, 0.01, true); % ��16��16��������ǿ

% ��ȷģʽ�����������
O_precise = CLAHE_16to8(I, 32, 0.005, false); % ��32��32��������

% ��ʾ�Ա�
figure;
subplot(1,3,1); imshow(I, []); title('ԭʼ16λ����');
subplot(1,3,2); imshow(O_fast); title('����ģʽ��8λ��');
subplot(1,3,3); imshow(O_precise); title('��ȷģʽ��8λ��');

function O = CLAHE_16to8(I, block_size, clip_limit, fast_mode)
    % I: 16λ����ͼ��uint16��0 - 65535��
    % block_size: ���С��Ĭ��16��
    % clip_limit: �ü���ֵ��ռ�����ر�����Ĭ��0.01��
    % fast_mode: �Ƿ����ģʽ��Ĭ��true��

    %% 1. Ԥ�����Գ����
    pad = floor(block_size/2);
    I_pad = padarray(I, [pad pad], 'symmetric', 'both'); % �Գ����

    %% 2. �ֿ����ӳ���16λ��8λ��
    [H, W] = size(I_pad);
    map = cell(H/block_size, W/block_size); % �洢ÿ�����ӳ���65536���ȣ�

    for i = 1:block_size:H
        for j = 1:block_size:W
            block = I_pad(i:i+block_size-1, j:j+block_size-1);
            hist = calc_hist(block, 65536); % ����65536 binsֱ��ͼ
            hist = clip_hist(hist, clip_limit, block_size); % �ü�
            cdf = normalize_cdf(hist); % ����16��8ӳ���
            map{(i-1)/block_size+1, (j-1)/block_size+1} = cdf;
        end
    end

    %% 3. ��̬��Χѹ��+��ֵ
    if fast_mode
        O = fast_interpolate(I, map, block_size, pad);
    else
        O = precise_interpolate(I, map, block_size, pad);
    end

    O = uint8(max(min(O, 255), 0)); % �ضϵ�0 - 255
end

function hist = calc_hist(block, bins)
    % ���룺2D�飨uint16��0 - 65535��
    hist = histcounts(block(:), bins, 'Normalization', 'probability');
end

function hist = clip_hist(hist, clip, block_size)
    % �ü���ֵ��clipΪ�����صı�������0.01 = 1%��
    max_cnt = clip * block_size^2;
    excess = sum(hist > max_cnt);
    hist = min(hist, max_cnt);
    hist = hist + excess / numel(hist); % ���ȷ����ʣ����
end

function cdf = normalize_cdf(hist)
    % ����65536���ȵ�ӳ���ÿ��16λֵ��Ӧһ��8λֵ��
    cdf = cumsum(hist);
    cdf = (cdf - cdf(1)) / (cdf(end) - cdf(1)) * 255;
    cdf = uint8(cdf);
end

function O = fast_interpolate(I, map, block, pad)
    [H, W] = size(I);
    O = zeros(H, W, 'uint8');

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
            v11 = map{i1,j1}(val16 + 1); % 16λ����+1��MATLAB��1��ʼ��
            v12 = map{i1,j2}(val16 + 1);
            v21 = map{i2,j1}(val16 + 1);
            v22 = map{i2,j2}(val16 + 1);

            % ˫���Բ�ֵȨ��
            dx = i - i1; dy = j - j1;
            weight = [(1 - dx)*(1 - dy), (1 - dx)*dy, dx*(1 - dy), dx*dy];

            % ת��Ϊ�������ͽ��м���
            weight_double = double(weight);
            v_double = double([v11; v12; v21; v22]);

            % ˫���Բ�ֵ
            O(y,x) = uint8(round(weight_double * v_double));
        end
    end
end

function O = precise_interpolate(I, map, block, pad)
    [H, W] = size(I);
    O = zeros(H, W, 'uint8');
    block_num = size(map);

    for y = 1:H
        for x = 1:W
            val16 = I(y,x);
            if val16 == 0 || val16 == 65535 % �������äԪ
                O(y,x) = median(I(max(y - 1,1):min(y + 1,H), max(x - 1,1):min(x + 1,W)));
                continue;
            end

            % �ҵ���������飨���9����
            i_center = (y + pad - 1) / block;
            j_center = (x + pad - 1) / block;
            i_start = max(1, floor(i_center - 0.5));
            i_end = min(block_num(1), ceil(i_center + 0.5));
            j_start = max(1, floor(j_center - 0.5));
            j_end = min(block_num(2), ceil(j_center + 0.5));

            total_weight = 0;
            sum_val = 0;
            n_times = 0;
            for i = i_start:i_end
                for j = j_start:j_end
                    % ���������꣨����䣩
                    block_y = (i - 1)*block + pad;
                    block_x = (j - 1)*block + pad;

                    % ����Ȩ�أ���˹��
                    dist = sqrt((y - block_y)^2 + (x - block_x)^2);
                    weight = exp(-(dist^2)/(2*(block/2)^2)); % ����Ȩ�ظ�

                    % ���
                    sum_val = sum_val + map{i,j}(val16 + 1) * weight;
                    total_weight = total_weight + weight;
                    n_times = n_times + 1;
                end
            end
            
            %fprintf(" %d" ,n_times);

            O(y,x) = uint8(round(sum_val / total_weight));
        end
        
        fprintf("\n");
    end
end     