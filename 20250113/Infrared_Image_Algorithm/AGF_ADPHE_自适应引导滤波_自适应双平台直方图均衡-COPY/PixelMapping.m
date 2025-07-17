%�򿪵�λ����
%��߶���У��˫ֱ��ͼ���⻯����ͼ����ǿ

function [newPixelVal,validNum] = PixelMapping(hist ,dimImage)
    frequency = hist;
    accumulation = zeros(1,16384); %14bitsλ��
    
    %Ѱ����Сֵ�ĺ�ѡ��
     lowValleyQuency = [];
     lowValleyIndex = [];
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %�Ƿ񲻵���0
            lowValleyQuency = [lowValleyQuency, frequency(i)]; % ����Ϊ0��Ԫ����ӵ�������
            lowValleyIndex = [lowValleyIndex ,i];
        end
    end

    %��ֵ�˲�����
    filter_windows_size = 64;
    frequency_filtered = mean_filter_1d(lowValleyQuency, filter_windows_size);
    lowValleyQuency_filtered = round(frequency_filtered);
    
    index = 1;
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %�Ƿ񲻵���0
            frequency(i) = lowValleyQuency_filtered(index);
            index = index + 1;
        end
    end
    
    %������Сֵ
    n = 15;
    localValley = []; %�洢�ֲ����ֵ

    % POLAR��Ϊ��������ʧ�ܵ�ʱ��
  
    for i = 1:length(lowValleyQuency_filtered) - n + 1
        % ȷ����ǰ�����ڵ�����
        window = lowValleyQuency_filtered(i:i + n - 1);
        % ��ȡ�����м�λ�õ�����������ȡ����ȷ��Ϊ����������
        middleIndex = floor(n / 2 + 1 / 2);

        % �жϴ����м�λ�õ�Ԫ���Ƿ��Ǵ����ڵ����ֵ
        if window(middleIndex) == min(window) 
             localValley = [localValley ,lowValleyIndex(i+middleIndex-1)];
             fprintf(" %d " ,lowValleyIndex(i+middleIndex-1));
        end
    end
    fprintf("\n�ֲ���Сֵ ��ȸ��� %d \n" , length(localValley));
    
    %���Ѱ�ҷָ��
    max_variance = 0;
    best_threshold = 0;
    L = length(frequency);
    avg_all = 0;
    for m = 1:L
        if frequency(m) == 0
            continue;
        end
        avg_all = avg_all + m * frequency(m);
    end
    for t = 1:L - 1
        if frequency(t) == 0
            continue;
        end
        sum1 = 0;
        mu1 = 0;
        for m = 1:t
            if frequency(m) == 0
                continue;
            end
            sum1 = sum1 + frequency(m);
            mu1 = mu1 + m * frequency(m);
        end
        sum2 = 0;
        mu2 = 0;
        for m = t + 1:L
            if frequency(m) == 0
                continue;
            end
            sum2 = sum2 + frequency(m);
            mu2 = mu2 + m * frequency(m);
        end
        w0 = sum1 / (sum1+sum2);
        w1 = 1 - w0;
        
        variance = w0 * (mu1/sum1 - (avg_all / (sum1+sum2)))^2 + w1 * (mu2/sum2 - (avg_all / (sum1+sum2)))^2;
        if variance > max_variance
            max_variance = variance;
            best_threshold = t;
        end
    end
    fprintf("OSTU��ֵ %d \n" ,best_threshold);
    
    %�������µĶ�����ֵ
    draw2threshold = 0;
    for i = 1:length(localValley)-1
        if best_threshold >= localValley(i) && best_threshold <= localValley(i+1)
            if best_threshold - localValley(i) < localValley(i+1) - best_threshold
                draw2threshold = localValley(i);
            else
                draw2threshold = localValley(i+1);
            end
            break;
        end
    end
    fprintf("����OSTU��ֵ %d \n" ,draw2threshold);
    if draw2threshold == 0
        draw2threshold = best_threshold;
    end
    fprintf("���º�OSTU��ֵ %d \n" ,draw2threshold);
   
    index = draw2threshold;
    
    %�ƽ�ָ��
    frequency = frequency /sum(frequency(:));
    
    E = 0;
    for i = 1 : length(frequency)
        E = E + i * frequency(i);
    end
    
    L = length(frequency);
    E = E / L;
    fprintf("\nEֵ��0.5�� %.3f \n" ,E);
    
    alpha = 0;
    if E <= 0.5
        alpha = E;
    else
        alpha = 1-E;
    end
    
    alpha = 0.5;
  
    h = hist;
    %��Сֵ
    %minValue = min(h);
    minValue = Inf;
    for i = 1:index
        if h(i) ~= 0 && h(i) < minValue
            minValue = h(i);
        end
    end
    maxValue = max(h(1:index));
    
    hc = zeros(1,L);
    
    omiga1=1-index/L;
    omiga2=1;
    omiga3=1+index/L;
    
    omiga1=1;
    omiga2=1;
    omiga3=1;
    
    for i = 1 : index
        if h(i) == 0
            continue;
        end
        val = (h(i)-minValue)/(maxValue -minValue);
        hc(i) = 1/3 *(val^(alpha * omiga1) +val^(alpha * omiga2) + val^(alpha * omiga3));
        hc(i) = val^(alpha * omiga1);
    end
    
    
    minValue = Inf;
    for i = index+1:length(h)
        if h(i) ~= 0 && h(i) < minValue
            minValue = h(i);
        end
    end
    maxValue = max(h(index+1:length(h)));
    
    omiga1=1-(L-index)/L;
    omiga2=1;
    omiga3=1+(L-index)/L;
    
    omiga1=1;
    omiga2=1;
    omiga3=1;
    
    for i = index+1:length(h)
        if h(i) == 0
            continue;
        end
        val = (h(i)-minValue)/(maxValue -minValue);
        hc(i) = 1/3 *(val^(alpha * omiga1) +val^(alpha * omiga2) + val^(alpha * omiga3));
        hc(i) = val^(alpha * omiga1);
    end

    non_zero_count_left = sum(hc(1:index)~=0);
    non_zero_count_right = sum(hc(index+1:L)~=0);
    transition_width = min(non_zero_count_left ,non_zero_count_right) * 0.05; % ��������Ŀ�ȣ����Ե���
    %transition_width = 10;
%     for i = 1:transition_width
%         weight = i / transition_width;
% %         frequency(draw2threshold - i + 1) = (1 - weight) * frequency(draw2threshold - i + 1) + weight * frequency(draw2threshold+1);
% %         frequency(draw2threshold+i) = weight * frequency(draw2threshold - i + 1) + (1 - weight) * frequency(draw2threshold+i);
%     %         frequency(draw2threshold - i + 1) = frequency(draw2threshold - i + 1) * 4.0;
%     %         frequency(draw2threshold+i) = frequency(draw2threshold+i)  * 4.0; 
%         hc(draw2threshold - i + 1) = hc(draw2threshold - i + 1) * 2.0;
%         hc(draw2threshold+i) = hc(draw2threshold+i)  * 2.0; 
%     end
%     for i = 1:transition_width
%     weight = i / transition_width;
%         hc(draw2threshold - i + 1) = (1 - weight) * hc(draw2threshold - i + 1) + weight * hc(draw2threshold+1);
%         hc(draw2threshold+i) = weight * hc(draw2threshold - i + 1) + (1 - weight) * hc(draw2threshold+i);
%     end
    
    
    pc = zeros(1,L);
    sum_val= sum(hc(1:index));
    for i = 1 : index
        pc(i) = hc(i)/sum_val;
    end
    sum_val= sum(hc(index+1:L));
    for i = index+1 : L
        pc(i) = hc(i)/sum_val;
    end
    
    cc = zeros(1,L);
    cc(1,1) = pc(1);
    for i = 2 : index
        cc(i) = cc(i-1) + pc(i);
    end
    cc(1,index + 1) = pc(index + 1);
    for i = index + 2 : L
        cc(i) = cc(i-1) + pc(i);
    end
    
    f = zeros(1,L);
%     for i = 1 : index
%         f(i) = index* 0.6 * (cc(i) - 0.5*pc(i));
%     end
%     for i = index+1 : L
%         f(i) = (index+1) * 0.6 + (L-index-2)*1.2 * (cc(i) - 0.5*pc(i));
%     end
%     newPixelVal = uint8(round(round(f)/64.0));

    t1 = sum(frequency(1:index)~=0);
    t2 = sum(frequency(index+1:L)~=0);
%     t1 = sum(frequency(1:index));
%     t2 = sum(frequency(index+1:L));
    
    %f_gray = t1/(t1+t2)*255-20;
    f_gray = t1/(t1+t2)*255-20;
    b_gray = 255 - f_gray;
    
    for i = 1 : index
        f(i) = f_gray * (cc(i) - 0.5*pc(i));
    end
    for i = index+1 : L
        f(i) = f_gray + b_gray * (cc(i) - 0.5*pc(i));
    end
    newPixelVal = uint8(round(f));

    fprintf("done\n");
    
    frequency = hist;
    % �������ɷ���ЧԪ�صĸ��ʺ���
    newFrequency = []; % ��ʼ��������Ϊ��
    for i = 1:length(frequency)
        if frequency(i) ~= 0 %�Ƿ񲻵���0
            newFrequency = [newFrequency, frequency(i)]; % ����Ϊ0��Ԫ����ӵ�������
        end
    end
    % ͼ��ĻҶȼ���
    validNum = length(newFrequency);
    
end


function filtered_data = mean_filter_1d(data, window_size)
    % ��ȡ���ݳ���
    data_length = length(data);
    % ��ʼ���˲��������
    filtered_data = zeros(1, data_length);
    % ���㴰�ڰ뾶�����贰�ڴ�СΪ������
    half_window = floor(window_size / 2);
    
    % �������ݽ����˲�����
    for i = 1:data_length
        % ���ڴ洢���������ݵĺ�
        sum_value = 0;
        % ȷ����ǰԪ�ض�Ӧ���˲����ڷ�Χ
        start_idx = max(1, i - half_window);
        end_idx = min(data_length, i + half_window);
        % ���㴰�������ݵĺ�
        for j = start_idx:end_idx
            sum_value = sum_value + data(j);
        end
        % ���㴰�������ݵ�ƽ��ֵ������ֵ���˲��������
        filtered_data(i) = sum_value / (end_idx - start_idx + 1);
    end
end
