function [correct_image ] = BadPixelDetection1(GrayImage ,input_dir ,name)

    %盲元阈值
    threhold_up = 80;
    threhold_down = 30;
    bad_pixel_num = 0;

    %盲元表
    bad_pixel_sheet = zeros(size(GrayImage));

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
            if (p1>threhold_down && p2>threhold_down && p3>threhold_down && p4>threhold_down && p5>threhold_down && p6>threhold_down && p7>threhold_down && p8>threhold_down) 
    %          if (p1>threhold_down && p2>threhold_down && p3>threhold_down && p4>threhold_down && p5>threhold_down && p6>threhold_down && p7>threhold_down && p8>threhold_down) ...
    %              &&(p1<threhold_up && p2<threhold_up && p3<threhold_up && p4<threhold_up && p5<threhold_up && p6<threhold_up && p7<threhold_up && p8<threhold_up)

                %fprintf("i%d  j%d " ,i ,j);

                bad_pixel_sheet(i, j) = 255;

                bad_pixel_num = bad_pixel_num +1 ;

                %盲元校正
                pixle_val0 = GrayImage(i,j);    
                pixle_val = fix_bad_pixel(GrayImage ,i ,j);    

                GrayImage(i,j) = pixle_val;
                %GrayImage(i,j) = 0;
                %fprintf(" 盲元%d  校正%d\n",pixle_val0 ,pixle_val);
            end
        end
    end
    fprintf("盲元总数 % d\n",bad_pixel_num);

    save_temp_dir = strcat(input_dir ,name ,"盲元表.png");
    save_temp_dir = char(save_temp_dir(1));
    imwrite(uint8(bad_pixel_sheet), save_temp_dir);
    correct_image = GrayImage;
end

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