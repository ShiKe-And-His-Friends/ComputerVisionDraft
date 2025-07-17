clc;
clear;

cols = 640;
rows = 512;

input_dir = "D:\MATLAB_CODE\BadPixelDetection\";
name = "x";

%盲元阈值
threhold = 30;
bad_pixel_num = 0;

fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;

%保存拉伸后的数据
min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));
fprintf("max %d min %d \n" ,max_val ,min_val);

%保存图片
save_raw_image = uint8(round((GrayImage-min_val)/(max_val-min_val)*255));

save_temp_dir = strcat(input_dir ,name ,"原始图.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(save_raw_image, save_temp_dir);

%盲元表
bad_pixel_sheet = zeros(size(GrayImage));

%权重值
pixel_weight_sheet = zeros(size(GrayImage));

%搜索盲元
[rows, cols] = size(GrayImage);
for i = 3:rows-2
    for j = 3:cols-2
        neighborhood = GrayImage(i-1:i+1, j-1:j+1);
        center = neighborhood(2,2);
        p1 = abs(neighborhood(1,1)-center);
        p2 = abs(neighborhood(1,2)-center);
        p3 = abs(neighborhood(1,3)-center);
        p4 = abs(neighborhood(2,1)-center);
        p5 = abs(neighborhood(2,3)-center);
        p6 = abs(neighborhood(3,1)-center);
        p7 = abs(neighborhood(3,2)-center);
        p8 = abs(neighborhood(3,3)-center);
        
        %组成P数组
        p_number = [p1 ,p2 ,p3 ,p4 ,p5 ,p6 ,p7 ,p8];
        max_value = max(p_number(:));
        min_value = min(p_number(:));
        remaining_p_numbers = [];
        for index = 1:length(p_number)
            if p_number(index) ~= max_value && p_number(index) ~= min_value
                remaining_p_numbers = [remaining_p_numbers, p_number(index)];
            end
        end
        
        z1 = abs(neighborhood(1,1)-neighborhood(3,3));
        z2 = abs(neighborhood(1,2)-neighborhood(3,2));
        z3 = abs(neighborhood(1,3)-neighborhood(3,1));
        z4 = abs(neighborhood(2,1)-neighborhood(2,3));
        %组成Z数组
        z_number = [z1 ,z2 ,z3 ,z4 ];
        max_value = max(z_number(:));
        remaining_z_numbers = [];
        for index = 1:length(z_number)
            if z_number(index) ~= max_value
                remaining_z_numbers = [remaining_z_numbers, z_number(index)];
            end
        end
        
         v1 = sum(remaining_p_numbers(:));
         v2 = sum(remaining_z_numbers(:));
         value = v1/v2;
         
         if value ~= inf && value > 70
             pixel_weight_sheet(i,j) = 255;
             bad_pixel_num = bad_pixel_num +1 ;
         end
        
%         if p1>threhold && p2>threhold && p3>threhold && p4>threhold && p5>threhold && p6>threhold && p7>threhold && p8>threhold
%             fprintf("i%d  j%d " ,i ,j);
%             bad_pixel_sheet(i, j) = 255;
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
    end
end
fprintf("盲元总数 % d\n",bad_pixel_num);

save_temp_dir = strcat(input_dir ,name ,"盲元表.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(uint8(bad_pixel_sheet), save_temp_dir);

%保存图片
save_raw_image = uint8(round((GrayImage-min_val)/(max_val-min_val)*255));

save_temp_dir = strcat(input_dir ,name ,"校正图.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(save_raw_image, save_temp_dir);


%保存图片
%save_raw_image = uint8(round((pixel_weight_sheet-min_val)/(max_val-min_val)*255));
save_raw_image = uint8(round(pixel_weight_sheet));

save_temp_dir = strcat(input_dir ,name ,"权值图.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(save_raw_image, save_temp_dir);

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