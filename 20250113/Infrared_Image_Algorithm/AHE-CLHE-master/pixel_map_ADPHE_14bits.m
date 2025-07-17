function newPixelVal = pixel_map_ADPHE_14bits(hist,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384);
    
    % 重新生成非无效元素的概率函数
    newFrequency = []; % 初始化新数组为空
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %是否不等于0
            newFrequency = [newFrequency, frequency(i)]; % 将不为0的元素添加到新数组
        end
    end
    
    %搜索POLAR峰值
    n = 5;%窗口大小
    localPolar = zeros(1, length(newFrequency)); %存储局部最大值
    localPolar_idx = 1;
    
    for i = 1:length(newFrequency) - n + 1
        % 确定当前窗口内的数据
        window = newFrequency(i:i + n - 1);
        % 获取窗口中间位置的索引（向下取整，确保为整数索引）
        middleIndex = floor(n / 2 + 1 / 2);
        % 判断窗口中间位置的元素是否是窗口内的最大值
        if window(middleIndex) == max(window)
             % 如果是最大值，
             localPolar(1,localPolar_idx) = max(window);
             localPolar_idx = localPolar_idx+1;
        end
    end
    %localPolar
    
    %TODO 全增处理
    
    % 设置上平台值
    T_up = sum(localPolar)/(localPolar_idx-1);
    
    % 设置下平台值
    %T_down = min(dimImage(1)*dimImage(2) ,T_up *length(newFrequency)) / 16384.0;
    T_down = length(newFrequency)/ 256.0;
    disp("up_limit = " + T_up);
    disp("down_limit = " + T_down);
    
    %修改概率密度函数
    for i = 1:length(frequency)
        if frequency(i) ~= 0
            if frequency(i) > T_up
                frequency(i) = T_up;
            elseif frequency(i) < T_down
                frequency(i) = T_down;
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
  
   % ADPHE双平台自适应直方图均衡
    newPixelVal = uint8(255 * (accumulation - minValue) / (maxValue - minValue));
    
    