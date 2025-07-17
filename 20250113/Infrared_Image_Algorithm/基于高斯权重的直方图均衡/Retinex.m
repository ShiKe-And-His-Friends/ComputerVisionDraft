% https://blog.csdn.net/m0_46256255/article/details/133862194
% Retinexʵ��ͼ��ȥ��
% ���������
%  f����ͼ�����

% ���������
%  In�������ͼ��

% ����·���������ļ�
clc;clear;close all;
f = imread( 'D:\Document\��ֵ���500+ͼ������\test\dump\����3-AGF_ADPHE-�ں�ǰ�Ļ���ͼ.png');
fr = f(:, :);
%�������͹�һ��
mr = mat2gray(im2double(fr));
%����alpha����
%alpha = randi([80 100], 1)*20;
alpha = randi([80 100], 1)*100;

%����ģ���С
n = floor(min([size(f, 1) size(f, 2)])*0.5);
%��������
n1 = floor((n+1)/2);
for i = 1:n
    for j = 1:n
        %��˹����
        b(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*alpha))/(pi*alpha);
    end
end
%����˲�
nr1 = imfilter(mr,b,'conv', 'replicate');
ur1 = log(nr1);
tr1 = log(mr);
yr1 = (tr1-ur1)/3;
%����beta����
beta = randi([80 100], 1)*1;
%����ģ���С
x = 32;
for i = 1:n
    for j = 1:n
        %��˹����
        a(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*beta))/(6*pi*beta);
    end
end
%����˲�
nr2 = imfilter(mr,a,'conv', 'replicate');
ur2 = log(nr2);
tr2 = log(mr);
yr2 = (tr2-ur2)/3;
%����eta����
eta = randi([80 100], 1)*200;
for i = 1:n
    for j = 1:n
        %��˹����
        e(i,j)  = exp(-((i-n1)^2+(j-n1)^2)/(4*eta))/(4*pi*eta);
    end
end
%����˲�
nr3 = imfilter(mr,e,'conv', 'replicate');
ur3 = log(nr3);
tr3 = log(mr);
yr3 = (tr3-ur3)/3;
dr = yr1+yr2+yr3;
cr = im2uint8(dr);
% ���ɴ����ķ����õ����ͼ��
In = cr;
%�����ʾ
figure;
subplot(2, 2, 1); imshow(f); title('ԭͼ��', 'FontWeight', 'Bold');
subplot(2, 2, 2); imshow(In); title('������ͼ��', 'FontWeight', 'Bold');
% �ҶȻ������ڼ���ֱ��ͼ
Q  = f;
M = In;
subplot(2, 2, 3); imhist(Q, 64); title('ԭ�Ҷ�ֱ��ͼ', 'FontWeight', 'Bold');
subplot(2, 2, 4); imhist(M, 64); title('�����ĻҶ�ֱ��ͼ', 'FontWeight', 'Bold');
imwrite(In, 'D:\Document\��ֵ���500+ͼ������\test\dump\retinex.png');
