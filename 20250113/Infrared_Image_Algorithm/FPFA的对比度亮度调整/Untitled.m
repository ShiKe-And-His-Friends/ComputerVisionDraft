clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\Users\shike\Desktop\dde_14\";
save_dir = "C:\Users\shike\Desktop\dde_14_result\";
name = "x";

% input_dir = "C:\Picture\���ݲɼ�-20250330\19ms\14bit��ͼ���������ɼ�)\19ms-14bit- 2025-03-30 10-40-54\";
% save_dir = "C:\Picture\���ݲɼ�-20250330\find_max_result_gama\";
% name = "19ms-14bit- 2025-03-30 10-40-54";

fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;



fileID = fopen(save_dir + name + "-����.raw", 'wb');
fwrite(fileID, GrayImage, 'uint16'); 
fclose(fileID);

min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));
fprintf("max %d min %d \n" ,max_val ,min_val);

global_max = max(GrayImage(:));
[rows, cols] = size(GrayImage);
[row_index, col_index] = find(GrayImage == global_max);
fprintf("������� %d \n" ,global_max);
fprintf("���� %d %d \n" ,row_index, col_index);

save_raw_image = uint8(round((GrayImage-min_val)/(max_val-min_val)*255));

outputPath = 'C:\Users\shike\Desktop\dde_14_result\a3.txt';
% ��������˳�򱣴�Ϊ�ı��ļ���ÿ��һ��ֵ��
fileID = fopen(outputPath, 'w');
if fileID == -1
    error('�޷���������ļ���');
end
matrix = save_raw_image;
% �������ȱ�������д���ļ�
for row = 1:size(matrix, 1)
    for col = 1:size(matrix, 2)
        fprintf(fileID, '%04X\n', matrix(row, col));  % ����6λС��
    end
end
fclose(fileID);
disp(['�����Ѱ�������˳�򱣴浽: ', outputPath]);

save_temp_dir = strcat(save_dir ,name ,"8λͼ-��������.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(uint8(save_raw_image), save_temp_dir);

% ��ȡͼ��
img = save_raw_image; % ʹ��MATLAB�Դ��Ĳ���ͼ��

img = rot90(img, -1);

% ���öԱȶȺ����Ȳ���
contrast = 80;   % �ԱȶȲ��� (-100��100֮��)
bright = 0;     % ���Ȳ��� (-255��255֮��)

% Ӧ�öԱȶ���ǿ
[enhancedImg1, enhancedImg2] = contrastEnhancement(img, contrast, bright);

% ��ʾ���
figure('Position', [100, 100, 1200, 300]);

subplot(1, 3, 1);
imshow(img);
title('ԭʼͼ��');
axis on;

subplot(1, 3, 2);
imshow(enhancedImg1);
title('��һ����ǿ����');
axis on;

subplot(1, 3, 3);
imshow(enhancedImg2);
title('�ڶ�����ǿ����');
axis on;

% ��ʾ��ǿ������Ч������
figure('Position', [100, 100, 800, 500]);
x = linspace(0, 1, 100);
y1 = zeros(size(x));
y2 = zeros(size(x));
avg = mean(img(:));

for i = 1:length(x)
    y1(i) = avg + 100 * (x(i) - avg) / (100 - contrast) + bright/255;
    y2(i) = avg + (x(i) - avg) * (100 + contrast) / 100 + bright/255;
end

plot(x, x, 'k--', 'LineWidth', 1); % ԭʼӳ����
hold on;
plot(x, y1, 'r', 'LineWidth', 2); % ��һ����ǿ����
plot(x, y2, 'b', 'LineWidth', 2); % �ڶ�����ǿ����
plot(avg, avg, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g'); % ƽ���Ҷȵ�

title('�Աȶ���ǿ����ӳ������');
xlabel('ԭʼ�Ҷ�ֵ');
ylabel('��ǿ��Ҷ�ֵ');
legend('ԭʼӳ��', '��һ������', '�ڶ�������', 'ƽ���Ҷ�');
grid on;

rgb_image = repmat(save_raw_image, [1, 1, 3]);
rgb_image = paint_image(rgb_image,row_index, col_index);
save_temp_dir = strcat(save_dir ,name ,"8λͼ-���ֵλ��.png");
save_temp_dir = char(save_temp_dir(1));
rgb_image = rot90(rgb_image, -1);
imwrite(uint8(rgb_image), save_temp_dir);


function out = paint_image(image ,i,j)
    image(i, j ,1) = 255;
    image(i, j ,2) = 0;
    image(i, j ,3) = 0;
    image(i-1, j ,1) = 255;
    image(i-1, j ,2) = 0;
    image(i-1, j ,3) = 0;
    image(i+1, j ,1) = 255;
    image(i+1, j ,2) = 0;
    image(i+1, j ,3) = 0;
    
    image(i, j-1 ,1) = 255;
    image(i, j-1 ,2) = 0;
    image(i, j-1 ,3) = 0;
    image(i-1, j-1 ,1) = 255;
    image(i-1, j-1 ,2) = 0;
    image(i-1, j-1 ,3) = 0;
    image(i+1, j-1 ,1) = 255;
    image(i+1, j-1 ,2) = 0;
    image(i+1, j-1 ,3) = 0;
    
    image(i, j+1 ,1) = 255;
    image(i, j+1 ,2) = 0;
    image(i, j+1 ,3) = 0;
    image(i-1, j+1 ,1) = 255;
    image(i-1, j+1 ,2) = 0;
    image(i-1, j+1 ,3) = 0;
    image(i+1, j+1 ,1) = 255;
    image(i+1, j+1 ,2) = 0;
    image(i+1, j+1 ,3) = 0;
    out = image;
end


function min_nonzero = find_min_nonzero(A)
    min_nonzero = Inf; % ��ʼ��Ϊ�����
    [rows, cols] = size(A);
    for i = 1:rows
        for j = 1:cols
            if A(i, j) ~= 0 && A(i, j) < min_nonzero
                min_nonzero = A(i, j);
            end
        end
    end
    if min_nonzero == Inf
        % ���û�з���Ԫ�أ����� NaN �������������Ϊ����ֵ
        min_nonzero = NaN; 
    end
end


function [enhancedImg1, enhancedImg2] = contrastEnhancement(img, contrast, bright)
    % �Աȶ���ǿ����
    % ����:
    %   img - ����Ҷ�ͼ��
    %   contrast - �ԱȶȲ��� (-100��100֮��)
    %   bright - ���Ȳ���
    % ���:
    %   enhancedImg1 - ��һ����ǿ����������ͼ��
    %   enhancedImg2 - �ڶ�����ǿ����������ͼ��
    
    % ȷ������ͼ��Ϊdouble��������[0,1]��Χ��
    if ~isa(img, 'double')
        img = im2double(img);
    end
    
    % ����ƽ���Ҷ�
    average_gray = mean(img(:));
    
    % ʵ�ֵ�һ���Աȶ���ǿ����
    % Gray = Average_Gray + 100 * (Gray-Average_Gray)/(100-Contrast) + Bright
    enhancedImg1 = average_gray + 100 * (img - average_gray) / (100 - contrast) + bright/255;
    
    % ʵ�ֵڶ����Աȶ���ǿ����
    % Gray = Average_Gray + (Gray-Average_Gray) * (100 + Contrast)/100 + Bright
    enhancedImg2 = average_gray + (img - average_gray) * (100 + contrast) / 100 + bright/255;
    
    % �ü���[0,1]��Χ
    enhancedImg1 = max(min(enhancedImg1, 1), 0);
    enhancedImg2 = max(min(enhancedImg2, 1), 0);
end