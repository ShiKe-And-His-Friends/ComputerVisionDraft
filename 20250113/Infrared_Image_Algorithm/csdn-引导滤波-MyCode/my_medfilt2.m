function filtered_image = my_medfilt2(image, filter_size)
    [rows, cols] = size(image);
    pad_size = floor(filter_size(1)/2); % ����߽�����С��������贰���������Σ�����չΪ�������
    padded_image = padarray(image, [pad_size, pad_size], 'symmetric'); % ��ͼ����б߽����
    filtered_image = zeros(rows, cols); % ��ʼ���˲����ͼ��

    for i = 1:rows
        for j = 1:cols
            % ��ȡ�Ե�ǰ����Ϊ���ĵľֲ������ڵ�����
            window = padded_image(i:i + 2 * pad_size, j:j + 2 * pad_size);
            % ������������չƽΪһά����������
            sorted_window = sort(window(:));
            % ȡ�м�ֵ��Ϊ�˲����ֵ
            filtered_image(i, j) = sorted_window(ceil(length(sorted_window)/2));
        end
    end
    filtered_image = uint8(filtered_image); % �����ת��Ϊ���ʵ��������ͣ��������ԭͼ��Ϊuint8���ͣ�
end