% 2024.12.23 sk95120
% ����Ӧ˫ƽֱ̨��ͼ ADPHE

clear all;
clc;

rows = 640;
cols = 512;
fid = fopen('D:\Document\��ֵ���500+ͼ������\test\dump\����1-AGF_ADPHE-�ֲ����ͼ.raw', 'r');
rawData = fread(fid, 640*512, 'uint16');
fclose(fid);
GrayImage = reshape(rawData, rows, cols);

%8bitͼ�軬λ
%GrayImage = GrayImage - 16384;
fileID = fopen('D:\����5.raw', 'wb');
fwrite(fileID, GrayImage, 'uint16'); % ����ʵ���������͵���
fclose(fileID);

%ԭʼ��14bitsֱ��ͼ
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
title('ԭ�Ҷ�ͼ');
subplot(1,2,2);
imshow(CLHEGrayImage);
title('�Աȶ�����ֱ��ͼ���⻯�Ҷ�ͼ');
figure(12);
subplot(1,2,1);
bar(0:16383,newHist);
axis([0 16383 0 max(max(newHist),clip*cols*rows+100)]);
hold on
plot([0,16383],[clip*cols*rows clip*cols*rows],'r')
title('ԭ�Ҷ�ͼhist');
subplot(1,2,2);
bar(0:16383,clipHist);
axis([0 16383 0 max(max(clipHist),clip*cols*rows+100)]);
hold on
plot([0,16383],[clip*cols*rows clip*cols*rows],'r')
title('�Աȶ�����ֱ��ͼ���⻯�Ҷ�ͼhist');

imwrite(CLHEGrayImage, 'D:\Document\��ֵ���500+ͼ������\test\����.png');
