function [newPixelVal,validNum] = PixelMapping(hist ,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384); %14bitsλ��
    
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
        % �жϴ����м�λ�õ�Ԫ���Ƿ��Ǵ����ڵ����ֵ
        if window(middleIndex) == max(window)
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
    T_down = min(T_down_b,T_down_c);
    
    disp("��ƽֵ̨ up_limit = " + T_up);
    %fprintf("%f %f %f ",T_down_a ,T_down_b,T_down_c); %��ӡ��ƽֵ̨
    disp("��ƽֵ̨ down_limit = " + T_down);
    
    % ���λ�
    T_up_uint = round(T_up);
    T_down_uint = round(T_down);
    
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
    validNum = length(newFrequency);
    fprintf("�Ҷȼ��� %d \n" ,validNum);
    
    % ADPHE˫ƽ̨����Ӧֱ��ͼ����
    newPixelVal = uint8(round(255.0 * (accumulation - minValue) / (maxValue - minValue)));

end
 