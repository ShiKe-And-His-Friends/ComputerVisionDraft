% ͼ����ֱ��ͼ���� + DDEϸ����ǿ
% ���ߣ�AI����
% �汾��1.0
% ���ڣ�2025-07-10

clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\MATLAB_CODE\input_image\";
name = "����1";

%äԪ��ֵ
threhold_up = 80;
threhold_down = 20;
bad_pixel_num = 0;

fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

GrayImage2 = rot90(GrayImage,-1);
GrayImage = GrayImage2;

inputImage = GrayImage;
% ��ȡ����ͼ�񣨼���Ϊ14λ�Ҷ�ͼ��ʹ��16λ�洢��
%%inputImage = imread('����1-ADPHE.png'); % �滻Ϊʵ��ͼ��·��
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
enhancedImage = ddeEnhancement(equaledImage,inputImage, DDE_Level, H_ALL, V_ALL);

% ��ʾ�ͱ�����
figure;
subplot(1,3,1); imshow(inputImage, []); title('ԭʼͼ��');
subplot(1,3,2); imshow(equaledImage, []); title('ֱ��ͼ����');
subplot(1,3,3); imshow(enhancedImage, []); title('DDE��ǿ��');
imwrite(enhancedImage, 'enhanced_image.png');

% ------------------------- �������� ------------------------- %

% ����1��ֱ��ͼ����
function equaledImage = histogramEqualization(image, H_ALL, V_ALL, MAP_Max, MAP_Mid, MAP_Min)
    [rows, cols] = size(image);
    % ��ʼ��ֱ��ͼͳ��
    histMap = zeros(16384, 1); % ����8λ�Ҷ�
    
    % ͳ��ֱ��ͼ
    for i = 1:rows
        for j = 1:cols
            pixel = round(image(i,j)) + 1; % ת��Ϊ8λ�Ҷ�
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
            pixel = round(image(i,j)) + 1;
            newPixel = MAP_Min + (MAP_Max - MAP_Min) * cdf(pixel);
            equaledImage(i,j) = newPixel / 255; % ��һ���� [0,1]
        end
    end
end

% ����2��DDEϸ����ǿ
function enhancedImage = ddeEnhancement(image8u, raw_image, DDE_Level, H_ALL, V_ALL)
    [rows, cols] = size(raw_image);
    enhancedImage = zeros(size(raw_image));
    
    % �������򴰿ڣ�5x5��
    windowSize = 5;
    halfSize = floor(windowSize / 2); % ǿ��ת��Ϊ����
    
    % ���ͼ��
    padImage = padarray(raw_image, [halfSize halfSize], 'replicate');
    padImage8u = padarray(image8u, [halfSize halfSize], 'replicate');
    
    % ����ÿ������
    for i = halfSize + 1 : rows + halfSize
        for j = halfSize + 1 : cols + halfSize
            % ��ȡ5x5����
            window = padImage(i - halfSize : i + halfSize, ...
                              j - halfSize : j + halfSize);
                          
            window8u = padImage8u(i - halfSize : i + halfSize, ...
                              j - halfSize : j + halfSize);
            
            % Step 1: ������������ֵ
            centerPixel = window(halfSize + 1, halfSize + 1);
            centerPixel8u = window8u(halfSize + 1, halfSize + 1);
            
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
            %enhancedPixel = lowFreq + DDE_Level / 100 * highFreq;
            
            enhancedPixel = centerPixel8u + DDE_Level / 100 * highFreq;
            
            % Step 8: ���ʹ���
            enhancedPixel = max(0, min(1, enhancedPixel));
            
            % д����
            enhancedImage(i - halfSize, j - halfSize) = enhancedPixel;
        end
    end
end