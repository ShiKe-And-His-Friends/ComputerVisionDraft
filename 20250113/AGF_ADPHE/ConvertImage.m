%
%��ȡ��о����8bits��doubleС��rawͼ����ȥƫ�ã�ת����PNGͼ
cols = 640;
rows = 512;

fid = fopen('D:\Document\��ֵ���500+ͼ������\2024-12-13\����6-8bit\x.raw', 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);

%оƬ��rawͼ��Ҫ�ü�
GrayImage = GrayImage - 16384;

%ϸ��ͼת��
image_T = double(zeros(rows,cols));
for i=1:cols
     for j=1:rows
         image_T(j,i) = GrayImage(i,j);
    end
end

resultImage = uint8(image_T);
imwrite(resultImage, 'D:\Document\��ֵ���500+ͼ������\test\COMPARE\����6-�ں�.png');