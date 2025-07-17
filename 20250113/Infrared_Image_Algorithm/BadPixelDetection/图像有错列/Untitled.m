%ͼ�����

clc;
clear;

%��ȡ14bits��doubleС��rawͼ
cols = 640;
rows = 512;

input_dir = "C:\Users\shike\Desktop\�½��ļ���\";

fid = fopen(input_dir  + "0.bin", 'r');
rawData1 = fread(fid, rows*cols, 'uint16');
fclose(fid);

fid = fopen(input_dir  + "1.bin", 'r');
rawData2 = fread(fid, rows*cols, 'uint16');
fclose(fid);

GrayImage1 = reshape(rawData1,cols ,rows);
GrayImage2 = reshape(rawData2,cols ,rows);

%оƬ��rawͼ
GrayImage1 = GrayImage1 - 16384;
GrayImage2 = GrayImage2 - 16384;

row_means1 = mean(GrayImage1, 2);
row_means2 = mean(GrayImage2, 2);

x = 1:640; % ���ɴ� 1 �� 10 ��һά����
y1 = row_means1';
y2 = row_means2';

n= 4;
%y2 = circular_right_shift(y2, n);

% ������������
figure; % ����һ���µ�ͼ�δ���
plot(x, y1, 'b-o', 'DisplayName', '00'); % ���� y1 ���飬��ɫ������Բ�α��
hold on; % ���ֵ�ǰͼ�Σ��Ա���ͬһͼ�л��ƶ������
plot(x, y2, 'r-s', 'DisplayName', '11'); % ���� y2 ���飬��ɫ���������α��

% ��ӱ���ͱ�ǩ
title('��λ'); % ����ͼ�α���
xlabel('x'); % ���� x ���ǩ
ylabel('y'); % ���� y ���ǩ

% ���ͼ��
legend; % ��ʾͼ�������� DisplayName ������ʾ��������

% ��ʾ������
grid on; % ��ʾ�����ߣ�����۲����ݵ�

result = mean_absolute_difference(y1, y2);
disp(['���������ƽ�����Բ�ֵΪ: ', num2str(result)]);



function mad = mean_absolute_difference(arr1, arr2)
    % �����������ĳ����Ƿ���ͬ
    if length(arr1) ~= length(arr2)
        error('��������ĳ��ȱ�����ͬ');
    end
    % �����ӦԪ�صĲ�ֵ
    diff = arr1 - arr2;
    % ȡ��ֵ�ľ���ֵ
    abs_diff = abs(diff);
    % ����ƽ�����Բ�ֵ
    mad = mean(abs_diff);
end


function shifted_array = circular_left_shift(arr, n)
    % ��ȡ���鳤��
    len = length(arr);
    % ���� n �������鳤�ȵ����
    n = mod(n, len);
    % ����ѭ������
    shifted_array = [arr(n+1:end), arr(1:n)];
end

function shifted_array = circular_right_shift(arr, n)
    % ��ȡ����ĳ���
    len = length(arr);
    % ���� n �������鳤�ȵ����
    n = mod(n, len);
    % ����ѭ�����Ʋ���
    shifted_array = [arr(end - n + 1:end), arr(1:end - n)];
end

