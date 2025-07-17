%保存raw图
fileID = fopen('D:\Document\均值相差500+图像数据\test\1.raw', 'wb');
fwrite(fileID, GrayImage, 'uint16'); % 根据实际数据类型调整
fclose(fileID);

GrayImage = GrayImage / 64;
GrayImage = uint8(GrayImage);
imwrite(GrayImage, 'D:\Document\均值相差500+图像数据\test\1.png');