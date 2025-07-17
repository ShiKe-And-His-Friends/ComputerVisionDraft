function newPixelVal = pixel_map_ADPHE_14bits(hist,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384);
    
    % �������ɷ���ЧԪ�صĸ��ʺ���
    newFrequency = []; % ��ʼ��������Ϊ��
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %�Ƿ񲻵���0
            newFrequency = [newFrequency, frequency(i)]; % ����Ϊ0��Ԫ����ӵ�������
        end
    end
    
    %����POLAR��ֵ
    n = 5;%���ڴ�С
    localPolar = zeros(1, length(newFrequency)); %�洢�ֲ����ֵ
    localPolar_idx = 1;
    
    for i = 1:length(newFrequency) - n + 1
        % ȷ����ǰ�����ڵ�����
        window = newFrequency(i:i + n - 1);
        % ��ȡ�����м�λ�õ�����������ȡ����ȷ��Ϊ����������
        middleIndex = floor(n / 2 + 1 / 2);
        % �жϴ����м�λ�õ�Ԫ���Ƿ��Ǵ����ڵ����ֵ
        if window(middleIndex) == max(window)
             % ��������ֵ��
             localPolar(1,localPolar_idx) = max(window);
             localPolar_idx = localPolar_idx+1;
        end
    end
    %localPolar
    
    %TODO ȫ������
    
    % ������ƽֵ̨
    T_up = sum(localPolar)/(localPolar_idx-1);
    
    % ������ƽֵ̨
    %T_down = min(dimImage(1)*dimImage(2) ,T_up *length(newFrequency)) / 16384.0;
    T_down = length(newFrequency)/ 256.0;
    disp("up_limit = " + T_up);
    disp("down_limit = " + T_down);
    
    %�޸ĸ����ܶȺ���
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
    % HEֱ��ͼ���⻯
    for i = 2:16384
        accumulation(1,i) = accumulation(1,i-1) + frequency(1,i);
    end

    % �ҳ������е���Сֵ
    minValue = min(accumulation);
    % �ҳ������е����ֵ
    maxValue = max(accumulation);
  
   % ADPHE˫ƽ̨����Ӧֱ��ͼ����
    newPixelVal = uint8(255 * (accumulation - minValue) / (maxValue - minValue));
    
    