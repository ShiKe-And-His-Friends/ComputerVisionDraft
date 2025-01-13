% 评价图片质量
function  EvaluateQuality(Image)

    [rows,cols] = size(Image);

    %计算图片图片的信噪比SNR
    % 计算图像像素的平均值
    image_mean = mean(Image(:));
    % 计算信号功率
    signal_power = image_mean^2;
    % 计算噪声功率
    noise_power = mean(mean((Image - image_mean).^2));
    % 计算信噪比（单位：dB）
    SNR = 10 * log10(signal_power / noise_power);
    disp(['    图像的μ均值 = ', num2str(image_mean), ' RMS均方根 = ' ,num2str(sqrt(noise_power))]);
    disp(['    图像的SNR信噪比 = ', num2str(SNR), ' dB']);
    
    %计算图片的平均梯度AG
    % 使用Sobel算子计算水平方向的梯度
    Gx = double(imfilter(Image, fspecial('sobel')));
    % 使用Sobel算子计算垂直方向的梯度
    Gy = double(imfilter(Image, fspecial('sobel').'));
    % 计算梯度幅值
    gradient_magnitude = sqrt(Gx.^2 + Gy.^2);
    % 计算平均梯度
    average_gradient = mean(gradient_magnitude(:));
    disp(['    图像的AG平均梯度 = ', num2str(average_gradient)]);
    
    %计算图像的全局对比度
    % 计算图像的均值
    image_mean = mean(double(Image(:)));
    % 计算图像的标准差
    image_std = std(double(Image(:)));
    % 计算对比度
    contrast = image_std / image_mean;
    disp(['    图像的对比度 =  ', num2str(contrast)]);
    
    %计算图像信息熵
    % 获取图像的灰度直方图，返回每个灰度值的像素数量
    histogram = imhist(Image);
    % 计算图像的总像素数量
    total_pixels = numel(Image);
    % 计算每个灰度值出现的概率
    probabilities = histogram / total_pixels;
    % 计算信息熵
    entropy = 0;
    for i = 1:length(probabilities)
        if probabilities(i) > 0
            entropy = entropy - probabilities(i) * log2(probabilities(i));
        end
    end
    disp("    信息熵 = " + entropy);
end

% 计算PSNR峰值信噪比
function  EvaluateQualityPSNR(original_image,processed_image)
    % 计算均方误差（MSE）
    [m, n] = size(original_image);
    MSE = sum(sum((original_image - processed_image).^2)) / (m * n);
    
    % 计算PSNR
    if MSE == 0
        PSNR = Inf;
    else
        PSNR = 10 * log10((255^2) / MSE);
    end
    disp(['图像的PSNR为: ', num2str(PSNR), 'dB']);
    
end
