% ͼ����ֱ��ͼ���� + DDEϸ����ǿ��14λ�Ҷȣ�
% ���ߣ�AI����
% �汾��1.1
% ���ڣ�2025-07-10

clc;
clear;

% ͼ����ֱ��ͼ���� + DDEϸ����ǿ��14λ�Ҷȣ�
% ���ߣ�AI����
% �汾��1.1
% ���ڣ�2025-07-10

% ��ȡ����ͼ�񣨼���Ϊ14λ�Ҷ�ͼ��ʹ��16λ�洢��
inputImage = imread('����1-ADPHE.png'); % �滻Ϊʵ��ͼ��·��
inputImage = im2double(inputImage); % ת��Ϊ˫���ȸ����� [0,1]

% �������ã�����FPGA���������
H_ALL = 640;    % ˮƽ�ֱ���
V_ALL = 512;    % ��ֱ�ֱ���
MAP_Max = 165;  % ֱ��ͼӳ�����ֵ
MAP_Mid = 90;   % �м�ֵ
MAP_Min = 89;   % ��Сֵ
DDE_Level = 100;% DDE��ǿϵ��

% Step 1: ֱ��ͼ����
equaledImage = histogramEqualization(inputImage, H_ALL, V_ALL, MAP_Max, MAP_Mid, MAP_Min);

% Step 2: DDEϸ����ǿ
enhancedImage = ddeEnhancement(equaledImage, DDE_Level, H_ALL, V_ALL);

% ��ʾ�ͱ�����
figure;
subplot(1,3,1); imshow(inputImage, []); title('ԭʼͼ��');
subplot(1,3,2); imshow(equaledImage, []); title('ֱ��ͼ����');
subplot(1,3,3); imshow(enhancedImage, []); title('DDE��ǿ��');
imwrite(uint16(enhancedImage * 16383), 'enhanced_image_14bit.png'); % ����Ϊ14λͼ��

% ------------------------- �������� ------------------------- %

% ����1��ֱ��ͼ���⣨����14λ�Ҷȣ�
function equaledImage = histogramEqualization(image, H_ALL, V_ALL, MAP_Max, MAP_Mid, MAP_Min)
    [rows, cols] = size(image);
    % ��ʼ��ֱ��ͼͳ��
    histMap = zeros(16384, 1); % 14λ�Ҷ�
    
    % ͳ��ֱ��ͼ
    for i = 1:rows
        for j = 1:cols
            pixel = round(image(i,j) * 16383) + 1; % ת��Ϊ14λ����
            histMap(pixel) = histMap(pixel) + 1;
        end
    end
    
    % �����ۻ��ֲ����� (CDF)
    cdf = cumsum(histMap);
    cdf = cdf / max(cdf); % ��һ��
    
    % ӳ�䵽�·�Χ [MAP_Min, MAP_Max]
    equaledImage = zeros(size(image));
    for i = 1:rows
        for j = 1:cols
            pixel = round(image(i,j) * 16383) + 1;
            newPixel = MAP_Min + (MAP_Max - MAP_Min) * cdf(pixel);
            equaledImage(i,j) = newPixel / 16383; % ��һ���� [0,1]
        end
    end
end

% ����2��DDEϸ����ǿ
function enhancedImage = ddeEnhancement(image, DDE_Level, H_ALL, V_ALL)
    [rows, cols] = size(image);
    enhancedImage = zeros(size(image));
    
    % �������򴰿ڣ�5x5��
    windowSize = 5;
    padImage = padarray(image, [windowSize/2 windowSize/2], 'replicate');
    
    % ����ÿ������
    for i = windowSize/2 + 1:rows + windowSize/2
        for j = windowSize/2 + 1:cols + windowSize/2
            % ��ȡ5x5����
            window = padImage(i-windowSize/2:i+windowSize/2, j-windowSize/2:j+windowSize/2);
            
            % Step 1: ������������ֵ
            centerPixel = window(3,3);
            
            % Step 2: �����ݶȣ����Բ��죩
            gradients = abs(window - centerPixel);
            
            % Step 3: ����Ȩ�أ������ݶȣ�
            weights = 1 ./ (1 + gradients); % �򻯰�Ȩ��
            
            % Step 4: ��Ȩ���
            weightedSum = sum(sum(weights .* window));
            weightSum = sum(sum(weights));
            
            % Step 5: ��һ����Ƶ����
            lowFreq = weightedSum / weightSum;
            
            % Step 6: ��Ƶ�������������� - ��Ƶ��
            highFreq = centerPixel - lowFreq;
            
            % Step 7: Ӧ��DDE��ǿϵ��
            enhancedPixel = lowFreq + DDE_Level/100 * highFreq;
            
            % Step 8: ���ʹ���
            enhancedPixel = max(0, min(1, enhancedPixel));
            
            enhancedImage(i-windowSize/2, j-windowSize/2) = enhancedPixel;
        end
    end
end