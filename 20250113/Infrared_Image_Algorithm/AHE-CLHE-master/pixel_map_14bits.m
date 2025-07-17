function newPixelVal = pixel_map_14bits(hist,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384);
    accumulation(1,1)=frequency(1,1);
    % HE直方图均衡化
    for i = 2:16384
        accumulation(1,i) = accumulation(1,i-1) + frequency(1,i);
    end
   
    % 找出向量中的最小值
    minValue = min(accumulation);
    % 找出向量中的最大值
    maxValue = max(accumulation);
  
    % 直方图线性压缩
    %newPixelVal = uint8(255 * (accumulation - minValue) / (maxValue - minValue));
    
    % NAV 分段压缩
%     newPixelVal_1 = uint8(315 * (accumulation - minValue) / (maxValue - minValue));
%     newPixelVal_2 = uint8(255 * (accumulation - minValue) / (maxValue - minValue));
%     newPixelVal_3 = uint8(200 * (accumulation - minValue) / (maxValue - minValue));
    
    newPixelVal = uint8(round(255.0 * (accumulation - minValue) / (maxValue - minValue)));

end