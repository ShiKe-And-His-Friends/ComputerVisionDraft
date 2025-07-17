function [newPixelVal,validNum] = muliti_CLHE_PixelMapping(hist ,dimImage)
   
    frequency = hist;
    accumulation = zeros(1,16384); %14bitsλ��
    
    %��ֵ�˲�����
%     frequency_filtered = fpge_filter_nonzero(frequency);
%     frequency = frequency_filtered;
    
    % �������ɷ���ЧԪ�صĸ��ʺ���
    newFrequency = []; % ��ʼ��������Ϊ��
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %�Ƿ񲻵���0
            newFrequency = [newFrequency, frequency(i)]; % ����Ϊ0��Ԫ����ӵ�������
        end
    end
    disp("��Ч�Ҷȼ� = " + length(newFrequency))

    %����POLAR��ֵ
    n = 9;% ���ڴ�С 5 9 15
    localPolar = zeros(1, length(newFrequency)); %�洢�ֲ����ֵ
    localPolar_idx = 1;
    disp("POLAR �������ڴ�С = " + n)
    
    % POLAR��Ϊ��������ʧ�ܵ�ʱ��
  
    for i = 1:length(newFrequency) - n + 1
        % ȷ����ǰ�����ڵ�����
        window = newFrequency(i:i + n - 1);
        % ��ȡ�����м�λ�õ�����������ȡ����ȷ��Ϊ����������
        middleIndex = floor(n / 2 + 1 / 2);
        if  max(window) <= 20 
            continue;
        end
        % �жϴ����м�λ�õ�Ԫ���Ƿ��Ǵ����ڵ����ֵ
        if window(middleIndex) == max(window) && window(middleIndex+1) ~= max(window) && window(middleIndex-1) ~= max(window)
             % ��������ֵ��
             localPolar(1,localPolar_idx) = max(window);
             fprintf('%d ', localPolar(1,localPolar_idx) );
             localPolar_idx = localPolar_idx+1;
        end
    end
    fprintf('\n');
    disp("Polar��ֵ���� " + localPolar_idx);
    %localPolar
    
    %TODO ȫ��ǿ����
    
    % ������ƽֵ̨
    T_up = sum(localPolar)/(localPolar_idx-1);
    
    % ������ƽֵ̨
    T_down_a = dimImage(1)*dimImage(2) / 16384.0;
    T_down_b = T_up *length(newFrequency) / 16384.0;
    T_down_c = length(newFrequency)/ 256.0; %���пƴ����ƽ̨��ֵ���㷽��
    T_down = min(T_down_a,T_down_b);
    
    disp("��ƽֵ̨ up_limit = " + T_up);
    %fprintf("%f %f %f ",T_down_a ,T_down_b,T_down_c); %��ӡ��ƽֵ̨
    disp("��ƽֵ̨ down_limit = " + T_down);
    
    % ���λ�
    T_up_uint = round(T_up);
    T_down_uint = round(T_down);
    
    %�޸ĸ����ܶȺ���
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
    fprintf("��ȡ%d ,����%d\n",add_nums,substact_nums);
    
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
    % HEֱ��ͼ���⻯
    for i = 2:16384
        accumulation(1,i) = accumulation(1,i-1) + frequency(1,i);
    end

    % �ҳ������е���Сֵ
    %minValue = min(accumulation);
    minValue = Inf;
    for i = 1:length(accumulation)
        if accumulation(i) ~= 0 && accumulation(i) < minValue
            minValue = accumulation(i);
        end
    end
    % �ҳ������е����ֵ
    maxValue = max(accumulation);
  
    % ͼ��ĻҶȼ���
    validNum = length(newFrequency);
    fprintf("�Ҷȼ��� %d \n" ,validNum);
    
    %������Ч�Ҷȼ���
    base_threshold = 0;
    if validNum < 255
        base_threshold = floor((255 - validNum)/2.0);
    end
    
    % ADPHE˫ƽ̨����Ӧֱ��ͼ����
    newPixelVal = uint8(round((255.0-2.0*base_threshold) * (accumulation - minValue) / (maxValue - minValue)) + base_threshold);

end

function filtered_data = fpge_filter_nonzero(arr)
    % ��ȡ��������ĳ���
    data_length = length(arr);
    % ��ʼ���˲�������ݣ���ԭ���鳤����ͬ
    filtered_data = zeros(1, data_length);
    window_size = 9;
    % ���㴰�ڰ뾶�����ڴ�СΪ����ʱ��
    half_window = floor(window_size / 2);
    
    gaussian_kernel = [4 8 4 8 16 8 4 8 4];

    for i = 1 :half_window
        filtered_data(i) = arr(i);
    end
    for i = data_length - half_window :data_length
        filtered_data(i) = arr(i);
    end

    % �������������ÿһ��Ԫ��
    for i = half_window + 1:data_length - half_window -1
        if arr(i) == 0
           continue; 
        end
        % ���ڴ洢��ǰԪ�����˲��˾���Ľ�������˲����ֵ��
        sum_value = 0;
        sum_kernel = 0;
        % �����˲����뵱ǰԪ�ض�Ӧ��λ��
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
