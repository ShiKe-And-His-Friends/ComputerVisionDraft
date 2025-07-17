% 2024.12.23 sk95120
% 自适应双平台直方图 ADPHE

clear all;
clc;

rows = 640;
cols = 512;
fid = fopen('D:\Document\均值相差500+图像数据\test\dump\场景1-AGF_ADPHE-分层基底图.raw', 'r');
rawData = fread(fid, 640*512, 'uint16');
fclose(fid);
GrayImage = reshape(rawData, rows, cols);

%8bit图需滑位
%GrayImage = GrayImage - 16384;
fileID = fopen('D:\场景5.raw', 'wb');
fwrite(fileID, GrayImage, 'uint16'); % 根据实际数据类型调整
fclose(fileID);

%原始的14bits直方图
newHist = get_hist_14bits(GrayImage);

% CLHE
clip = 0.0025;
clipHist = CLHE_14bits(clip,GrayImage);

%HE
CLHEPixelMap_14bits = pixel_map_14bits(clipHist,size(GrayImage));

% ADPHE 
%CLHEPixelMap_14bits = pixel_map_ADPHE_14bits(clipHist,size(GrayImage));


CLHEGrayImage = uint8(zeros(rows,cols));
for i=1:rows
     for j=1:cols
        idx = GrayImage(i,j);
        CLHEGrayImage(i,j) = CLHEPixelMap_14bits(idx);
    end
end
figure(11);
subplot(1,2,1);
imshow(GrayImage/64);
title('原灰度图');
subplot(1,2,2);
imshow(CLHEGrayImage);
title('对比度限制直方图均衡化灰度图');
figure(12);
subplot(1,2,1);
bar(0:16383,newHist);
axis([0 16383 0 max(max(newHist),clip*cols*rows+100)]);
hold on
plot([0,16383],[clip*cols*rows clip*cols*rows],'r')
title('原灰度图hist');
subplot(1,2,2);
bar(0:16383,clipHist);
axis([0 16383 0 max(max(clipHist),clip*cols*rows+100)]);
hold on
plot([0,16383],[clip*cols*rows clip*cols*rows],'r')
title('对比度限制直方图均衡化灰度图hist');

imwrite(CLHEGrayImage, 'D:\Document\均值相差500+图像数据\test\场景.png');
