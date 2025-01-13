% ����ͼƬ����
function  EvaluateQuality(Image)

    [rows,cols] = size(Image);

    %����ͼƬͼƬ�������SNR
    % ����ͼ�����ص�ƽ��ֵ
    image_mean = mean(Image(:));
    % �����źŹ���
    signal_power = image_mean^2;
    % ������������
    noise_power = mean(mean((Image - image_mean).^2));
    % ��������ȣ���λ��dB��
    SNR = 10 * log10(signal_power / noise_power);
    disp(['    ͼ��Ħ̾�ֵ = ', num2str(image_mean), ' RMS������ = ' ,num2str(sqrt(noise_power))]);
    disp(['    ͼ���SNR����� = ', num2str(SNR), ' dB']);
    
    %����ͼƬ��ƽ���ݶ�AG
    % ʹ��Sobel���Ӽ���ˮƽ������ݶ�
    Gx = double(imfilter(Image, fspecial('sobel')));
    % ʹ��Sobel���Ӽ��㴹ֱ������ݶ�
    Gy = double(imfilter(Image, fspecial('sobel').'));
    % �����ݶȷ�ֵ
    gradient_magnitude = sqrt(Gx.^2 + Gy.^2);
    % ����ƽ���ݶ�
    average_gradient = mean(gradient_magnitude(:));
    disp(['    ͼ���AGƽ���ݶ� = ', num2str(average_gradient)]);
    
    %����ͼ���ȫ�ֶԱȶ�
    % ����ͼ��ľ�ֵ
    image_mean = mean(double(Image(:)));
    % ����ͼ��ı�׼��
    image_std = std(double(Image(:)));
    % ����Աȶ�
    contrast = image_std / image_mean;
    disp(['    ͼ��ĶԱȶ� =  ', num2str(contrast)]);
    
    %����ͼ����Ϣ��
    % ��ȡͼ��ĻҶ�ֱ��ͼ������ÿ���Ҷ�ֵ����������
    histogram = imhist(Image);
    % ����ͼ�������������
    total_pixels = numel(Image);
    % ����ÿ���Ҷ�ֵ���ֵĸ���
    probabilities = histogram / total_pixels;
    % ������Ϣ��
    entropy = 0;
    for i = 1:length(probabilities)
        if probabilities(i) > 0
            entropy = entropy - probabilities(i) * log2(probabilities(i));
        end
    end
    disp("    ��Ϣ�� = " + entropy);
end

% ����PSNR��ֵ�����
function  EvaluateQualityPSNR(original_image,processed_image)
    % ���������MSE��
    [m, n] = size(original_image);
    MSE = sum(sum((original_image - processed_image).^2)) / (m * n);
    
    % ����PSNR
    if MSE == 0
        PSNR = Inf;
    else
        PSNR = 10 * log10((255^2) / MSE);
    end
    disp(['ͼ���PSNRΪ: ', num2str(PSNR), 'dB']);
    
end
