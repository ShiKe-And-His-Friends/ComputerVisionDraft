clear all;
clc;

% RGBImage = imread('testHE_n1.jpg');
% GrayImage = ToGray(RGBImage);
% oldHist = get_hist(GrayImage);
% newGrayImage = AHE([4,4],GrayImage);

x = 640;
y = 512;
fid = fopen('D:\Document\��ֵ���500+ͼ������\2024-12-13\����4-8bit\x.raw', 'r');
rawData = fread(fid, x*y, 'uint16');
fclose(fid);
GrayImage = reshape(rawData, x, y);

%8bitͼ�軬λ
GrayImage = GrayImage - 16384;
GrayImage = uint8(GrayImage);

oldHist = get_hist(GrayImage);
newGrayImage = AHE([4,4],GrayImage);
newHist = get_hist(newGrayImage);

figure(11);
subplot(1,2,1);
imshow(GrayImage);
title('ԭ�Ҷ�ͼ');
subplot(1,2,2);
imshow(newGrayImage);
title('����Ӧֱ��ͼ���⻯�Ҷ�ͼ');
figure(12);
subplot(1,2,1);
bar(0:255,oldHist);
axis([0 255 0 max(oldHist)]);
title('ԭ�Ҷ�ͼhist');
subplot(1,2,2);
bar(0:255,newHist);
axis([0 255 0 max(newHist)]);
title('����Ӧֱ��ͼ���⻯�Ҷ�ͼhist');
