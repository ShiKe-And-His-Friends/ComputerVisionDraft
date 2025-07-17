function [newPixelVal,validNum] = PixelMapping(hist ,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384); %14bits位宽
    
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
        % 判断窗口中间位置的元素是否是窗口内的最大值
        if window(middleIndex) == max(window)
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
    T_down = min(T_down_b,T_down_c);
    
    disp("上平台值 up_limit = " + T_up);
    %fprintf("%f %f %f ",T_down_a ,T_down_b,T_down_c); %打印下平台值
    disp("下平台值 down_limit = " + T_down);
    
    % 整形化
    T_up_uint = round(T_up);
    T_down_uint = round(T_down);
    
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
    validNum = length(newFrequency);
    fprintf("灰度级数 %d \n" ,validNum);
    
    % ADPHE双平台自适应直方图均衡
    newPixelVal = uint8(round(255.0 * (accumulation - minValue) / (maxValue - minValue)));

end
 