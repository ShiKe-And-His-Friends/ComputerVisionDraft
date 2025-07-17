function filtered_image = BilateralFilter(image, window_size, sigma_d, sigma_r)
    % ��ȡͼ�������������
    [rows, cols] = size(image);
    % ��ʼ���˲����ͼ����ԭͼ���С��ͬ
    filtered_image = zeros(rows, cols);
    % ���㴰�ڰ뾶�����贰�ڴ�СΪ������
    half_window = floor(window_size / 2);
    
    % ����ͼ���ÿһ������
    for y = 1:rows
        for x = 1:cols
            % ���ڴ洢��Ȩ�ͣ��˲����ֵ��
            sum_weighted_value = 0;
            % ���ڴ洢Ȩ���ܺ�
            sum_weight = 0;
            % ������ǰ���ض�Ӧ���˲�����
            for i = max(1, y - half_window):min(rows, y + half_window)
                for j = max(1, x - half_window):min(cols, x + half_window)
                    % ����ռ����Ȩ��
                    dist = sqrt((i - y)^2 + (j - x)^2);
                    spatial_weight = exp(-dist^2 / (2 * sigma_d^2));
                    % ����Ҷ�������Ȩ��
                    intensity_diff = double(image(i, j)) - double(image(y, x));
                    range_weight = exp(-intensity_diff^2 / (2 * sigma_r^2));
                    % �����ۺ�Ȩ��
                    weight = spatial_weight * range_weight;
                    % �ۼӼ�Ȩ����ֵ��Ȩ���ܺ�
                    sum_weighted_value = sum_weighted_value + double(image(i, j)) * weight;
                    sum_weight = sum_weight + weight;
                end
            end
            % �����˲�������ص�ֵ
            filtered_image(y, x) = sum_weighted_value / sum_weight;
        end
    end
end