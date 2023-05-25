clc;
clear;
% 图像序列的基本信息
% 512行 640行 像素深度16bit
read_fold_name = 'C:\Users\s05559\Desktop';
write_folf_name = 'C:\Users\s05559\Desktop';
col=4320;
row=2000;

listfile = dir(fullfile(read_fold_name ,'*.raw'));
nn = length(listfile);

for ii =1:nn
	raw_file_name = listfile(ii).name;
	raw_file_name = fullfile(read_fold_name ,raw_file_name);
	bmp_file_name = char(string(write_folf_name) + string(num2str(ii)) + '.bmp');
	fid = fopen(raw_file_name ,'r');
	raw_image = fread(fid ,[col ,row] ,'uint8'); % 读取图片文件
	raw_image = raw_image';
	%imtool(A ,[]);
	fclose(fid);
	imwrite(uint8(raw_image) ,bmp_file_name);
end