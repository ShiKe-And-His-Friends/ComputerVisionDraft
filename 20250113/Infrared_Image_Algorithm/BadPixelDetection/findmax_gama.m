clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\Picture\小目标截图-20250325\";
save_dir = "C:\Picture\小目标截图-20250325\find_max_result5_gama\";
name = "19ms-14位 -2025-03-25 05-12-07";


fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

%GrayImage = PixelFix(GrayImage);

%保存拉伸后的数据
min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));

%搜索盲元
[rows, cols] = size(GrayImage);

min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));
fprintf("max %d min %d \n" ,max_val ,min_val);

% 找到二维数组的最大值
global_max = max(GrayImage(:));

% 找到最大值的坐标
[row_index, col_index] = find(GrayImage == global_max);
fprintf("最大像素 %d \n" ,global_max);
fprintf("坐标 %d %d \n" ,row_index, col_index);

%保存图片
%save_raw_image = uint8(round((GrayImage-min_val)/(max_val-min_val)*255));

%对数
% 归一化图像数据到 [0, 1]
normalized_image = (double(GrayImage) - min_val) / (max_val - min_val);

% 设置伽马值
gamma = 3.5; % 可根据需要调整，小于 1 增强亮度，大于 1 降低亮度

% 进行伽马校正
gamma_corrected = normalized_image.^gamma;

min_val = double(min(gamma_corrected(:)));
max_val = double(max(gamma_corrected(:)));

save_raw_image = uint8(round(gamma_corrected*255));

% 反归一化图像数据到原始范围
%save_raw_image = uint8(gamma_corrected * (max_val - min_val) + min_val);



% sorted_data = sort(GrayImage(:));
% 
% % 计算数据的前80%和后20%的分割点
% split_index = floor(length(sorted_data) * 0.8);
% 
% % 提取前80%和后20%的数据
% first_part = sorted_data(1:split_index);
% second_part = sorted_data(split_index + 1:end);
% 
% % 计算第一段线性拉伸的参数
% a1 = 51 / (max(first_part) - min(first_part));
% b1 = -a1 * min(first_part);
% 
% % 计算第二段线性拉伸的参数
% a2 = (255 - 52) / (max(second_part) - min(second_part));
% b2 = 52 - a2 * min(second_part);
% 
% % 对原始二维数组进行两段线性拉伸
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
% % 确保拉伸后的数据在0到255的范围内
% stretched_data(stretched_data < 0) = 0;
% stretched_data(stretched_data > 255) = 255;
% 
% % 将数据转换为uint8类型
% save_raw_image = uint8(stretched_data);


save_temp_dir = strcat(save_dir ,name ," 8位图-γ值",num2str(gamma),".png");
save_temp_dir = char(save_temp_dir(1));
save_raw_image2 = rot90(save_raw_image, 1);
imwrite(save_raw_image2, save_temp_dir);

% save_temp_dir = strcat(save_dir ,name ,"最大值位置-不处理盲元-1.png");
% save_temp_dir = char(save_temp_dir(1));
% bad_pixel_sheet1 = bad_pixel_sheet;
% bad_pixel_sheet2 = rot90(bad_pixel_sheet1, -1);
% imwrite(uint8(bad_pixel_sheet2), save_temp_dir);

% rgb_image = repmat(save_raw_image, [1, 1, 3]);
% rgb_image = paint_image(rgb_image,row_index, col_index);
% save_temp_dir = strcat(save_dir ,name ,"8位图-最大值位置.png");
% save_temp_dir = char(save_temp_dir(1));
% rgb_image = rot90(rgb_image, -1);
% imwrite(uint8(rgb_image), save_temp_dir);

function min_nonzero = find_min_nonzero(A)
    min_nonzero = Inf; % 初始化为无穷大
    [rows, cols] = size(A);
    for i = 1:rows
        for j = 1:cols
            if A(i, j) ~= 0 && A(i, j) < min_nonzero
                min_nonzero = A(i, j);
            end
        end
    end
    if min_nonzero == Inf
        % 如果没有非零元素，返回 NaN 或根据需求设置为其他值
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
        %z1方向
        gray_fixable_value = (A(i-1,j-1) + A(i+1 ,j+1))/2;
        
    elseif z2 <= z1 && z2 <= z3 && z2 <= z4
        %z2方向
        gray_fixable_value = (A(i-1,j+1) + A(i+1 ,j-1))/2;
        
    elseif z3 <= z1 && z3 <= z2 && z3 <= z4 
        %z3方向
        gray_fixable_value = (A(i,j-1) + A(i ,j+1))/2;
        
    else
        %z4方向
        gray_fixable_value = (A(i,j-2) + A(i ,j+2))/2;
        
    end
    gray_fixable_value= round(gray_fixable_value);
end

