function [newPixelVal,validNum] = PixelMapping(hist ,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384); %14bitsλ��
    
    %�����ֵ
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
    fprintf("��ֵ%f ��׼��%f\n",miu ,std_v);
    
    % ���λ�
    T_up_uint = round(miu - 3 * std_v);
    T_down_uint = round(1 * std_v);

    %�޸ĸ����ܶȺ���
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
    % HEֱ��ͼ���⻯
    for i = 2:16384
        accumulation(1,i) = accumulation(1,i-1) + frequency(1,i);
    end

    % �ҳ������е���Сֵ
    minValue = min(accumulation);
    % �ҳ������е����ֵ
    maxValue = max(accumulation);
  
    % ͼ��ĻҶȼ���
    newFrequency = []; % ��ʼ��������Ϊ��
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %�Ƿ񲻵���0
            newFrequency = [newFrequency, frequency(i)]; % ����Ϊ0��Ԫ����ӵ�������
        end
    end
    validNum = length(newFrequency);
    fprintf("�Ҷȼ��� %d \n" ,validNum);
    
    %������Ч�Ҷȼ���
    base_threshold = 0;
    if validNum < 255
        base_threshold = floor((255 - validNum)/2.0);
    end
    
    %ֱ��ͼ����
    newPixelVal = uint8(round((255.0-2.0*base_threshold) * (accumulation - minValue) / (maxValue - minValue)) + base_threshold);
    
        % ͼ��ĻҶȼ���
    validNum = length(newFrequency);
    fprintf("�Ҷȼ��� %d \n" ,validNum);
    
end
