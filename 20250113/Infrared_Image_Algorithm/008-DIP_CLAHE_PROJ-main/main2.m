
% mars = imread('D:\Document\��оͼ������-20250105\dump\05E�̴�������-1-ԭʼ-�ں�ǰ�Ļ���ͼ.png');
% 
% M1 = dos_clahe(mars,[8 8],0.01);
% figure(1);
% imshow(mars);
% figure(2);
% imshow(M1);
% 
% save_temp_dir = strcat("D:\MATLAB_CODE\CLAHE-Github\008-DIP_CLAHE_PROJ-main\"  ,"05E�̴�������-1-ԭʼ-�ں�ǰ�Ļ���ͼ.png");
% save_temp_dir = char(save_temp_dir(1));
% imwrite(uint8(M1), save_temp_dir);

cols = 640;
rows = 512;

fid = fopen("D:\Document\��оͼ������-20250105\dump\05F-3-30000-�ֲ����ͼ.raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);

%оƬ��rawͼ��Ҫ�ü�
mars = GrayImage;

M1 = dos_clahe(mars,[8 8],0.0008);

fileID = fopen("D:\Document\��оͼ������-20250105\dump\a.raw", 'wb');
fwrite(fileID, uint16(M1), 'uint16'); 
fclose(fileID);
