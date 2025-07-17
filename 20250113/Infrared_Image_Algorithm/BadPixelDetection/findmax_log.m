clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\Picture\���ݲɼ�-20250330\10ms\14bit��ͼ���������ɼ�)\2025-03-30 10-46-28\";
save_dir = "C:\Picture\���ݲɼ�-20250330\find_max_result_gama\";
name = "2025-03-30 10-46-28";


fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

%��������������
min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));

%����äԪ
[rows, cols] = size(GrayImage);

min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));
fprintf("max %d min %d \n" ,max_val ,min_val);

% �ҵ���ά��������ֵ
global_max = max(GrayImage(:));

% �ҵ����ֵ������
[row_index, col_index] = find(GrayImage == global_max);
fprintf("������� %d \n" ,global_max);
fprintf("���� %d %d \n" ,row_index, col_index);

image = double(GrayImage);

% ����任����ĳ���
min_val = double(min(image(:)));
max_val = double(max(image(:)));
% ��������任��������Сֵ
log_min = log(1 + min_val);
log_max = log(1 + max_val);
% ������������
scale = 255 / (log_max - log_min);

% ���ж����任��ӳ�䵽 0 - 255
log_transformed_image = uint8(scale * (log(1 + double(image)) - log_min));

save_raw_image = log_transformed_image;

save_temp_dir = strcat(save_dir ,name ," 8λͼ.png");
save_temp_dir = char(save_temp_dir(1));
save_raw_image2 = rot90(save_raw_image, 1);
imwrite(save_raw_image2, save_temp_dir);

% save_temp_dir = strcat(save_dir ,name ,"���ֵλ��-������äԪ-1.png");
% save_temp_dir = char(save_temp_dir(1));
% bad_pixel_sheet1 = bad_pixel_sheet;
% bad_pixel_sheet2 = rot90(bad_pixel_sheet1, -1);
% imwrite(uint8(bad_pixel_sheet2), save_temp_dir);

% rgb_image = repmat(save_raw_image, [1, 1, 3]);
% rgb_image = paint_image(rgb_image,row_index, col_index);
% save_temp_dir = strcat(save_dir ,name ,"8λͼ-���ֵλ��.png");
% save_temp_dir = char(save_temp_dir(1));
% rgb_image = rot90(rgb_image, -1);
% imwrite(uint8(rgb_image), save_temp_dir);

function min_nonzero = find_min_nonzero(A)
    min_nonzero = Inf; % ��ʼ��Ϊ�����
    [rows, cols] = size(A);
    for i = 1:rows
        for j = 1:cols
            if A(i, j) ~= 0 && A(i, j) < min_nonzero
                min_nonzero = A(i, j);
            end
        end
    end
    if min_nonzero == Inf
        % ���û�з���Ԫ�أ����� NaN �������������Ϊ����ֵ
        min_nonzero = NaN; 
    end
end

function out = paint_image(image ,i,j)
    image(i, j ,1) = 255;
    image(i, j ,2) = 0;
    image(i, j ,3) = 0;
    image(i-1, j ,1) = 255;
    image(i-1, j ,2) = 0;
    image(i-1, j ,3) = 0;
    image(i+1, j ,1) = 255;
    image(i+1, j ,2) = 0;
    image(i+1, j ,3) = 0;
    
    image(i, j-1 ,1) = 255;
    image(i, j-1 ,2) = 0;
    image(i, j-1 ,3) = 0;
    image(i-1, j-1 ,1) = 255;
    image(i-1, j-1 ,2) = 0;
    image(i-1, j-1 ,3) = 0;
    image(i+1, j-1 ,1) = 255;
    image(i+1, j-1 ,2) = 0;
    image(i+1, j-1 ,3) = 0;
    
    image(i, j+1 ,1) = 255;
    image(i, j+1 ,2) = 0;
    image(i, j+1 ,3) = 0;
    image(i-1, j+1 ,1) = 255;
    image(i-1, j+1 ,2) = 0;
    image(i-1, j+1 ,3) = 0;
    image(i+1, j+1 ,1) = 255;
    image(i+1, j+1 ,2) = 0;
    image(i+1, j+1 ,3) = 0;
    out = image;
end

function gray_fixable_value = fix_bad_pixel(A ,i ,j)
    z1 = abs(A(i-1,j-1) - A(i+1 ,j+1));
    z2 = abs(A(i-1,j+1) - A(i+1 ,j-1));
    z3 = abs(A(i,j-1) - A(i ,j+1));
    z4 = abs(A(i,j-2) - A(i ,j+2));
    if z1 <= z2 && z1 <= z3 && z1 <= z4 
        %z1����
        gray_fixable_value = (A(i-1,j-1) + A(i+1 ,j+1))/2;
        
    elseif z2 <= z1 && z2 <= z3 && z2 <= z4
        %z2����
        gray_fixable_value = (A(i-1,j+1) + A(i+1 ,j-1))/2;
        
    elseif z3 <= z1 && z3 <= z2 && z3 <= z4 
        %z3����
        gray_fixable_value = (A(i,j-1) + A(i ,j+1))/2;
        
    else
        %z4����
        gray_fixable_value = (A(i,j-2) + A(i ,j+2))/2;
        
    end
    gray_fixable_value= round(gray_fixable_value);
end

