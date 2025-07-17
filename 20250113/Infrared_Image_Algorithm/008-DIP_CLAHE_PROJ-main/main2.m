
% mars = imread('D:\Document\机芯图像数据-20250105\dump\05E短窗套自研-1-原始-融合前的基底图.png');
% 
% M1 = dos_clahe(mars,[8 8],0.01);
% figure(1);
% imshow(mars);
% figure(2);
% imshow(M1);
% 
% save_temp_dir = strcat("D:\MATLAB_CODE\CLAHE-Github\008-DIP_CLAHE_PROJ-main\"  ,"05E短窗套自研-1-原始-融合前的基底图.png");
% save_temp_dir = char(save_temp_dir(1));
% imwrite(uint8(M1), save_temp_dir);

cols = 640;
rows = 512;

fid = fopen("D:\Document\机芯图像数据-20250105\dump\05F-3-30000-分层基底图.raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);

%芯片的raw图需要裁剪
mars = GrayImage;

M1 = dos_clahe(mars,[8 8],0.0008);

fileID = fopen("D:\Document\机芯图像数据-20250105\dump\a.raw", 'wb');
fwrite(fileID, uint16(M1), 'uint16'); 
fclose(fileID);
