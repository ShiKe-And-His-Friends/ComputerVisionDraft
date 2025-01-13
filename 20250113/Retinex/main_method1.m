%���ڸ�˹Ȩ�ؾֲ�ֱ��ͼѹ��

% ��ȡͼ���ԻҶ�ͼ��Ϊ����
image = imread('D:\Document\��ֵ���500+ͼ������\test\dump\����4-AGF_ADPHE-�ں�ǰ�Ļ���ͼ.png');
if size(image, 3) > 1
    image = rgb2gray(image);
end
% ȷ���ֲ����ڴ�С�͸�˹��׼��
window_size = 3;
sigma = 0.5;

% ���㴰�ڰ뾶
half_window = floor(window_size / 2);
[x, y] = meshgrid(-half_window:half_window, -half_window:half_window);
gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));

% ��ȡͼ��ߴ�
[height, width] = size(image);
% ��ʼ��������ͼ��
processed_image = zeros(height, width);
for y = 1:height
    for x = 1:width
        % ȷ���ֲ����ڷ�Χ
        row_start = max(1, y - half_window);
        row_end = min(height, y + half_window);
        col_start = max(1, x - half_window);
        col_end = min(width, x + half_window);
        % ��ȡ�ֲ������ڵ�ͼ�����ݺͶ�Ӧ�ĸ�˹Ȩ��
        local_image = image(row_start:row_end, col_start:col_end);
        local_kernel = gaussian_kernel(1:(row_end - row_start + 1), 1:(col_end - col_start + 1));
        % �����ֲ���Ȩֱ��ͼ
        local_histogram = zeros(1, 256);
        for i = 1:size(local_image, 1)
            for j = 1:size(local_image, 2)
                gray_level = local_image(i, j) + 1;
                local_histogram(gray_level) = local_histogram(gray_level) + local_kernel(i, j);
            end
        end
        % �Ծֲ�ֱ��ͼ����ѹ���������Լ򵥵�ֱ��ͼ���⻯Ϊ����
        cumulative_histogram = cumsum(local_histogram);
        normalized_cumulative_histogram = cumulative_histogram / sum(local_histogram);
        for i = 1:size(local_image, 1)
            for j = 1:size(local_image, 2)
                old_gray_level = local_image(i, j) + 1;
                new_gray_level = round(normalized_cumulative_histogram(old_gray_level) * 255);
                processed_image(row_start + i - 1, col_start + j - 1) = new_gray_level;
            end
        end
    end
end

% ��ʾԭʼͼ��ʹ�����ͼ��
subplot(1, 2, 1);
imshow(image);
title('ԭʼͼ��');
subplot(1, 2, 1);
imshow(uint8(processed_image));
title('���ڸ�˹Ȩ�ؾֲ�ֱ��ͼѹ�����ͼ��');