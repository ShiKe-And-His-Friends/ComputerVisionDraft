clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\Picture\小目标截图-20250325\";
save_dir = "C:\Picture\小目标截图-20250325\find_max_result2\";
name = "14bit-12ms   2025-03-25 05-36-58";


fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

%保存拉伸后的数据
min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));

%保存图片
save_raw_image = uint8(round((GrayImage-min_val)/(max_val-min_val)*255));

save_temp_dir = strcat(save_dir ,name ,"原始图.png");
save_temp_dir = char(save_temp_dir(1));
save_raw_image2 = rot90(save_raw_image, -1);
imwrite(save_raw_image2, save_temp_dir);

%搜索盲元
[rows, cols] = size(GrayImage);

%盲元阈值
threhold_up = 80;
threhold_down = 20;
bad_pixel_num = 0;
% for i = 3:rows-2
%     for j = 3:cols-2
%         neighborhood = GrayImage(i-1:i+1, j-1:j+1);
%         center = neighborhood(2,2);
%         p1 = abs(neighborhood(1,1)-center);
%         p2 = abs(neighborhood(1,2)-center);
%         p3 = abs(neighborhood(1,3)-center);
%         p4 = abs(neighborhood(2,1)-center);
%         p5 = abs(neighborhood(2,3)-center);
%         p6 = abs(neighborhood(3,1)-center);
%         p7 = abs(neighborhood(3,2)-center);
%         p8 = abs(neighborhood(3,3)-center);
%         if (p1>threhold_down && p2>threhold_down && p3>threhold_down && p4>threhold_down && p5>threhold_down && p6>threhold_down && p7>threhold_down && p8>threhold_down) 
% %          if (p1>threhold_down && p2>threhold_down && p3>threhold_down && p4>threhold_down && p5>threhold_down && p6>threhold_down && p7>threhold_down && p8>threhold_down) ...
% %              &&(p1<threhold_up && p2<threhold_up && p3<threhold_up && p4<threhold_up && p5<threhold_up && p6<threhold_up && p7<threhold_up && p8<threhold_up)
%             fprintf("i%d  j%d " ,i ,j);
% 
%             
%             bad_pixel_num = bad_pixel_num +1 ;
%             
%             %盲元校正
%             pixle_val0 = GrayImage(i,j);    
%             pixle_val = fix_bad_pixel(GrayImage ,i ,j);    
%             
%             GrayImage(i,j) = pixle_val;
%             fprintf(" 盲元%d  校正%d\n",pixle_val0 ,pixle_val);
%         end
%     end
% end

min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));
fprintf("max %d min %d \n" ,max_val ,min_val);

%盲元表
bad_pixel_sheet = zeros(rows, cols,3 ,'uint8');

% 找到二维数组的最大值
global_max = max(GrayImage(:));

% 找到最大值的坐标
[row_index, col_index] = find(GrayImage == global_max);
fprintf("最大像素 %d \n" ,global_max);
fprintf("坐标 %d %d \n" ,row_index, col_index);
bad_pixel_sheet = paint_image(bad_pixel_sheet,row_index, col_index);

save_temp_dir = strcat(save_dir ,name ,"最大值位置-不处理盲元-1.png");
save_temp_dir = char(save_temp_dir(1));
bad_pixel_sheet1 = bad_pixel_sheet;
bad_pixel_sheet2 = rot90(bad_pixel_sheet1, -1);
imwrite(uint8(bad_pixel_sheet2), save_temp_dir);

rgb_image = repmat(save_raw_image, [1, 1, 3]);
rgb_image = paint_image(rgb_image,row_index, col_index);
save_temp_dir = strcat(save_dir ,name ,"最大值位置-不处理盲元-2.png");
save_temp_dir = char(save_temp_dir(1));
rgb_image = rot90(rgb_image, -1);
imwrite(uint8(rgb_image), save_temp_dir);

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

