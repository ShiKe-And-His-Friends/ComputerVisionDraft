% ��ȡͼ��
original_image = imread('D:\Document\��ֵ���500+ͼ������\test\COMPARE\����1-AGF_ADPHE-�ں�ǰ�Ļ���ͼ.png');
% ����BBHE����
enhanced_image = BBHE(original_image);
enhanced_image = uint8(round(enhanced_image));
% ��ʾԭʼͼ��ʹ�����ͼ��
subplot(1, 2, 1);
imshow(original_image);
title('ԭʼͼ��');
subplot(1, 2, 2);
imshow(enhanced_image);
title('BBHE�����ͼ��');

imwrite(enhanced_image, 'D:\Document\��ֵ���500+ͼ������\test\����.png');

function enhanced_image = BBHE(image)
% ����ɫͼ��ת��Ϊ�Ҷ�ͼ����������ǲ�ɫͼ��
if size(image, 3) > 1
    image = rgb2gray(image);
end

% ����ͼ���ƽ�����ȣ���ֵ��
mean_brightness = mean(image(:));

% ��ȡͼ��ĻҶ�ֱ��ͼ
[hist_counts, gray_levels] = imhist(image);

% �ҵ�ֱ��ͼ�зָ������λ�ã�����ƽ�����ȶ�Ӧ�ĻҶȼ���
cumulative_hist = cumsum(hist_counts);
total_pixels = numel(image);
split_index = find(cumulative_hist >= mean_brightness / 255.0 * total_pixels, 1);

% �ָ�ֱ��ͼΪ������
lower_hist = hist_counts(1:split_index);
upper_hist = hist_counts(split_index + 1:end);

% ���������ֵ��ۻ��ֲ�������CDF��
lower_cdf = cumsum(lower_hist) / sum(lower_hist);
upper_cdf = cumsum(upper_hist) / sum(upper_hist);

% ����ӳ���
mapping_table_lower = zeros(1, length(gray_levels));
mapping_table_upper = zeros(1, length(gray_levels));

for i = 1:length(gray_levels)
    if i <= split_index
        mapping_table_lower(i) = round(lower_cdf(i) * (split_index - 1));
    else
        mapping_table_upper(i) = round(upper_cdf(i - split_index) * (length(gray_levels) - split_index - 1)) + split_index;
    end
end

% �ϲ�ӳ���
mapping_table = mapping_table_lower;
mapping_table(split_index + 1:end) = mapping_table_upper(split_index + 1:end);

% ����ӳ�����лҶ�ֵӳ�䣬�õ���ǿ���ͼ��
enhanced_image = mapping_table(image+1);

end