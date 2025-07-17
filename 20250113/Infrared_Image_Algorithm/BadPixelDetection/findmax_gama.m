clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\Picture\СĿ���ͼ-20250325\";
save_dir = "C:\Picture\СĿ���ͼ-20250325\find_max_result5_gama\";
name = "19ms-14λ -2025-03-25 05-12-07";


fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

%GrayImage = PixelFix(GrayImage);

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

%����ͼƬ
%save_raw_image = uint8(round((GrayImage-min_val)/(max_val-min_val)*255));

%����
% ��һ��ͼ�����ݵ� [0, 1]
normalized_image = (double(GrayImage) - min_val) / (max_val - min_val);

% ����٤��ֵ
gamma = 3.5; % �ɸ�����Ҫ������С�� 1 ��ǿ���ȣ����� 1 ��������

% ����٤��У��
gamma_corrected = normalized_image.^gamma;

min_val = double(min(gamma_corrected(:)));
max_val = double(max(gamma_corrected(:)));

save_raw_image = uint8(round(gamma_corrected*255));

% ����һ��ͼ�����ݵ�ԭʼ��Χ
%save_raw_image = uint8(gamma_corrected * (max_val - min_val) + min_val);



% sorted_data = sort(GrayImage(:));
% 
% % �������ݵ�ǰ80%�ͺ�20%�ķָ��
% split_index = floor(length(sorted_data) * 0.8);
% 
% % ��ȡǰ80%�ͺ�20%������
% first_part = sorted_data(1:split_index);
% second_part = sorted_data(split_index + 1:end);
% 
% % �����һ����������Ĳ���
% a1 = 51 / (max(first_part) - min(first_part));
% b1 = -a1 * min(first_part);
% 
% % ����ڶ�����������Ĳ���
% a2 = (255 - 52) / (max(second_part) - min(second_part));
% b2 = 52 - a2 * min(second_part);
% 
% % ��ԭʼ��ά�������������������
% stretched_data = zeros(size(GrayImage));
% for i = 1:size(GrayImage, 1)
%     for j = 2:size(GrayImage, 2)
%         if GrayImage(i, j) <= max(first_part)
%             stretched_data(i, j) = a1 * GrayImage(i, j) + b1;
%         else
%             stretched_data(i, j) = a2 * GrayImage(i, j) + b2;
%         end
%     end
% end
% 
% % ȷ��������������0��255�ķ�Χ��
% stretched_data(stretched_data < 0) = 0;
% stretched_data(stretched_data > 255) = 255;
% 
% % ������ת��Ϊuint8����
% save_raw_image = uint8(stretched_data);


save_temp_dir = strcat(save_dir ,name ," 8λͼ-��ֵ",num2str(gamma),".png");
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

