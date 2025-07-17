function [newPixelVal,validNum] = PixelMapping(hist ,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384); %14bits位宽
    
    %计算均值
    sum_val = 0;
    sum_count = 0;
    for i = 1:length(frequency)
        if frequency(i) ~= 0
            sum_val = sum_val + frequency(i)*i;
            sum_count = sum_count + frequency(i);
        end
    end
    miu = sum_val/sum_count;
    
    sum_squared_diff = 0; 
    for i = 1:length(frequency)
        if frequency(i) ~= 0
            squared_diff = (i - miu)*(i-miu)*frequency(i);
            sum_squared_diff = sum_squared_diff + squared_diff;
        end
    end
    std_v = sqrt(sum_squared_diff / (sum_count - 1));
    fprintf("均值%f 标准差%f\n",miu ,std_v);
    
    % 整形化
    T_up_uint = round(miu - 3 * std_v);
    T_down_uint = round(1 * std_v);

    %修改概率密度函数
    for i = 1:length(frequency)
        if frequency(i) ~= 0
            if frequency(i) > T_up_uint
                frequency(i) = T_up_uint;
            elseif frequency(i) < T_down_uint
                frequency(i) = T_down_uint;
            end
        end
    end
    
    accumulation(1,1)=frequency(1,1);
    % HE直方图均衡化
    for i = 2:16384
        accumulation(1,i) = accumulation(1,i-1) + frequency(1,i);
    end

    % 找出向量中的最小值
    minValue = min(accumulation);
    % 找出向量中的最大值
    maxValue = max(accumulation);
  
    % 图像的灰度级数
    newFrequency = []; % 初始化新数组为空
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %是否不等于0
            newFrequency = [newFrequency, frequency(i)]; % 将不为0的元素添加到新数组
        end
    end
    validNum = length(newFrequency);
    fprintf("灰度级数 %d \n" ,validNum);
    
    %处理有效灰度集数
    base_threshold = 0;
    if validNum < 255
        base_threshold = floor((255 - validNum)/2.0);
    end
    
    %直方图均衡
    newPixelVal = uint8(round((255.0-2.0*base_threshold) * (accumulation - minValue) / (maxValue - minValue)) + base_threshold);
    
        % 图像的灰度级数
    validNum = length(newFrequency);
    fprintf("灰度级数 %d \n" ,validNum);
    
end
