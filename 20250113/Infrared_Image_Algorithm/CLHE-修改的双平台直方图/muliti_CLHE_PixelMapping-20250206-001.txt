function [newPixelVal,validNum] = muliti_CLHE_PixelMapping(hist ,dimImage)
   
    frequency = hist;
    accumulation = zeros(1,16384); %14bits位宽
    
    %均值滤波处理
%     frequency_filtered = fpge_filter_nonzero(frequency);
%     frequency = frequency_filtered;
    
    % 重新生成非无效元素的概率函数
    newFrequency = []; % 初始化新数组为空
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %是否不等于0
            newFrequency = [newFrequency, frequency(i)]; % 将不为0的元素添加到新数组
        end
    end
    disp("有效灰度级 = " + length(newFrequency))

    %搜索POLAR峰值
    n = 9;% 窗口大小 5 9 15
    localPolar = zeros(1, length(newFrequency)); %存储局部最大值
    localPolar_idx = 1;
    disp("POLAR 搜索窗口大小 = " + n)
    
    % POLAR因为窗口搜索失败的时候
  
    for i = 1:length(newFrequency) - n + 1
        % 确定当前窗口内的数据
        window = newFrequency(i:i + n - 1);
        % 获取窗口中间位置的索引（向下取整，确保为整数索引）
        middleIndex = floor(n / 2 + 1 / 2);
        if  max(window) <= 20 
            continue;
        end
        % 判断窗口中间位置的元素是否是窗口内的最大值
        if window(middleIndex) == max(window) && window(middleIndex+1) ~= max(window) && window(middleIndex-1) ~= max(window)
             % 如果是最大值，
             localPolar(1,localPolar_idx) = max(window);
             fprintf('%d ', localPolar(1,localPolar_idx) );
             localPolar_idx = localPolar_idx+1;
        end
    end
    fprintf('\n');
    disp("Polar峰值个数 " + localPolar_idx);
    %localPolar
    
    %TODO 全增强处理
    
    % 设置上平台值
    T_up = sum(localPolar)/(localPolar_idx-1);
    
    % 设置下平台值
    T_down_a = dimImage(1)*dimImage(2) / 16384.0;
    T_down_b = T_up *length(newFrequency) / 16384.0;
    T_down_c = length(newFrequency)/ 256.0; %华中科大的下平台阈值计算方法
    T_down = min(T_down_a,T_down_b);
    
    disp("上平台值 up_limit = " + T_up);
    %fprintf("%f %f %f ",T_down_a ,T_down_b,T_down_c); %打印下平台值
    disp("下平台值 down_limit = " + T_down);
    
    % 整形化
    T_up_uint = round(T_up);
    T_down_uint = round(T_down);
    
    %修改概率密度函数
    add_nums = 0;
    for i = 1:length(frequency)
        if frequency(i) ~= 0
            if frequency(i) > T_up_uint
                add_nums = add_nums + (frequency(i) - T_up_uint);
                frequency(i) = T_up_uint;
            end
        end
    end
    
    substact_nums = 0;
    for i = 1:length(frequency)
        if frequency(i) ~= 0
            if frequency(i) < T_down_uint
                substact_nums = substact_nums + (frequency(i) -T_down_uint);
                frequency(i) = T_down_uint;
            end
        end
    end
    fprintf("截取%d ,增加%d\n",add_nums,substact_nums);
    
    if(add_nums + substact_nums > 0 )
        num_gray_levels = length(newFrequency);
        excess_pixels = add_nums + substact_nums;
        excess_per_level = floor(excess_pixels / num_gray_levels);
        remainder = excess_pixels - excess_per_level * num_gray_levels;
        
        for i = 1:length(frequency)
            if frequency(i) ~= 0
                frequency(i) = frequency(i) + excess_per_level;
            end
        end
        
        for i = 1:length(frequency)
            if frequency(i) ~= 0 && remainder > 0
                frequency(i) = frequency(i) + 1;
                remainder = remainder - 1;
            end
        end
        
    end
    
    accumulation(1,1)=frequency(1,1);
    % HE直方图均衡化
    for i = 2:16384
        accumulation(1,i) = accumulation(1,i-1) + frequency(1,i);
    end

    % 找出向量中的最小值
    %minValue = min(accumulation);
    minValue = Inf;
    for i = 1:length(accumulation)
        if accumulation(i) ~= 0 && accumulation(i) < minValue
            minValue = accumulation(i);
        end
    end
    % 找出向量中的最大值
    maxValue = max(accumulation);
  
    % 图像的灰度级数
    validNum = length(newFrequency);
    fprintf("灰度级数 %d \n" ,validNum);
    
    %处理有效灰度集数
    base_threshold = 0;
    if validNum < 255
        base_threshold = floor((255 - validNum)/2.0);
    end
    
    % ADPHE双平台自适应直方图均衡
    newPixelVal = uint8(round((255.0-2.0*base_threshold) * (accumulation - minValue) / (maxValue - minValue)) + base_threshold);

end

function filtered_data = fpge_filter_nonzero(arr)
    % 获取输入数组的长度
    data_length = length(arr);
    % 初始化滤波后的数据，与原数组长度相同
    filtered_data = zeros(1, data_length);
    window_size = 9;
    % 计算窗口半径（窗口大小为奇数时）
    half_window = floor(window_size / 2);
    
    gaussian_kernel = [4 8 4 8 16 8 4 8 4];

    for i = 1 :half_window
        filtered_data(i) = arr(i);
    end
    for i = data_length - half_window :data_length
        filtered_data(i) = arr(i);
    end

    % 遍历输入数组的每一个元素
    for i = half_window + 1:data_length - half_window -1
        if arr(i) == 0
           continue; 
        end
        % 用于存储当前元素与滤波核卷积的结果（即滤波后的值）
        sum_value = 0;
        sum_kernel = 0;
        % 遍历滤波核与当前元素对应的位置
        for j = max(half_window + 1, i - half_window):min(data_length - half_window -1, i + half_window)
            index = j - (i - half_window);
            if arr(j) ~= 0
                sum_value = sum_value + arr(j) * gaussian_kernel(index + 1);
                sum_kernel = sum_kernel + gaussian_kernel(index + 1);
            end
        end
        if sum_kernel ~= 0
            filtered_data(i) = round(sum_value / sum_kernel);
        else
            filtered_data(i) =  0;
        end

    end
end
