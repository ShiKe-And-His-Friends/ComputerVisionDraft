clear all;
clc;

% RGBImage = imread('testHE_n1.jpg');
% GrayImage = ToGray(RGBImage);
% [rows,cols] = size(GrayImage);

x = 640;
y = 512;
rows = x;
cols = y;
fid = fopen('D:\Document\均值相差500+图像数据\2024-12-13\场景4-8bit\x.raw', 'r');
rawData = fread(fid, x*y, 'uint16');
fclose(fid);
GrayImage = reshape(rawData, x, y);

%8bit图需滑位
GrayImage = GrayImage - 16384;
GrayImage = uint8(GrayImage);

newHist = get_hist(GrayImage);

% CLHE
clip = 0.03;
clipHist = CLHE(clip,GrayImage);
CLHEPixelMap = pixel_map(clipHist,size(GrayImage));
CLHEGrayImage = uint8(zeros(rows,cols));
for i=1:rows
    for j=1:cols
        CLHEGrayImage(i,j) = CLHEPixelMap(GrayImage(i,j)+1);
    end
end
figure(11);
subplot(1,2,1);
imshow(GrayImage);
title('原灰度图');
subplot(1,2,2);
imshow(CLHEGrayImage);
title('对比度限制直方图均衡化灰度图');
figure(12);
subplot(1,2,1);
bar(0:255,newHist);
axis([0 255 0 max(max(newHist),clip*cols*rows+100)]);
hold on
plot([0,255],[clip*cols*rows clip*cols*rows],'r')
title('原灰度图hist');
subplot(1,2,2);
bar(0:255,clipHist);
axis([0 255 0 max(max(clipHist),clip*cols*rows+100)]);
hold on
plot([0,255],[clip*cols*rows clip*cols*rows],'r')
title('对比度限制直方图均衡化灰度图hist');
