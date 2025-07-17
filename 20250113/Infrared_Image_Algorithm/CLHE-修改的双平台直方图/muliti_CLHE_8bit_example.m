% ��ȡͼ��
img = imread('D:\Document\��ֵ���500+ͼ������\test\2025-01-18ͼƬ�Ա�\05E�̴�������-2-AGF_ADPHE-DDE���ں�ͼ.png');
%img = rgb2gray(img); % ����ǲ�ɫͼ��ת��Ϊ�Ҷ�ͼ��

% ���ö�ƽֵ̨
platforms = [20, 50, 100, 150, 200]; % ʾ��ƽֵ̨

% ����ͼ��ֱ��ͼ
histogram = imhist(img);

% ��ƽ̨����
num_platforms = length(platforms);
for i = 1:num_platforms + 1
    if i == 1
        start_g = 0;
        end_g = platforms(i) - 1;
    elseif i == num_platforms + 1
        start_g = platforms(i - 1);
        end_g = 255;
    else
        start_g = platforms(i - 1);
        end_g = platforms(i) - 1;
    end
    
    % ����������ƽֵ̨������򵥼���ÿ������ƽֵ̨��ͬ���ɸ������������
    platform_value = 50; % ʾ��ƽֵ̨
    
    % �������������������Ƿ񳬹�ƽֵ̨
    interval_hist = histogram(start_g + 1:end_g + 1);
    excess_pixels = sum(interval_hist) - platform_value * (end_g - start_g + 1);
    if excess_pixels > 0
        % ���·��䳬�����ֵ�����
        num_gray_levels = end_g - start_g + 1;
        excess_per_level = floor(excess_pixels / num_gray_levels);
        remainder = excess_pixels - excess_per_level * num_gray_levels;
        
        for j = 1:num_gray_levels
            interval_hist(j) = platform_value + excess_per_level;
        end
        % ��������
        for j = 1:remainder
            interval_hist(j) = interval_hist(j) + 1;
        end
        
        histogram(start_g + 1:end_g + 1) = interval_hist;
    end
end

% ֱ��ͼ���⻯
cumulative_hist = cumsum(histogram);
total_pixels = numel(img);
mapping = uint8(255 * (cumulative_hist - cumulative_hist(1)) / (total_pixels - cumulative_hist(1)));

% �Ҷȼ�ӳ��
enhanced_img = mapping(img + 1);

% ��ʾԭʼͼ�����ǿ���ͼ��
subplot(1,2,1);
imshow(img);
title('ԭʼͼ��');
subplot(1,2,2);
imshow(enhanced_img);
title('��ƽ̨����ֱ��ͼ������ͼ��');