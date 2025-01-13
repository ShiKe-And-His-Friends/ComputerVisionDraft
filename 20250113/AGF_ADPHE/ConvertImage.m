%
%读取机芯出的8bits的double小端raw图，减去偏置，转换成PNG图
cols = 640;
rows = 512;

fid = fopen('D:\Document\均值相差500+图像数据\2024-12-13\场景6-8bit\x.raw', 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);

%芯片的raw图需要裁剪
GrayImage = GrayImage - 16384;

%细节图转置
image_T = double(zeros(rows,cols));
for i=1:cols
     for j=1:rows
         image_T(j,i) = GrayImage(i,j);
    end
end

resultImage = uint8(image_T);
imwrite(resultImage, 'D:\Document\均值相差500+图像数据\test\COMPARE\场景6-黑盒.png');