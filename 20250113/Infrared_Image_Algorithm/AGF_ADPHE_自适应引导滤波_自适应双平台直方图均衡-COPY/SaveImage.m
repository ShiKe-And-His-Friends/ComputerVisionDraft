%����rawͼ
fileID = fopen('D:\Document\��ֵ���500+ͼ������\test\1.raw', 'wb');
fwrite(fileID, GrayImage, 'uint16'); % ����ʵ���������͵���
fclose(fileID);

GrayImage = GrayImage / 64;
GrayImage = uint8(GrayImage);
imwrite(GrayImage, 'D:\Document\��ֵ���500+ͼ������\test\1.png');