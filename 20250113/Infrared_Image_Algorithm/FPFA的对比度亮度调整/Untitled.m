clc;
clear;

cols = 640;
rows = 512;

input_dir = "C:\Users\shike\Desktop\dde_14\";
save_dir = "C:\Users\shike\Desktop\dde_14_result\";
name = "x";

% input_dir = "C:\Picture\数据采集-20250330\19ms\14bit截图（含连续采集)\19ms-14bit- 2025-03-30 10-40-54\";
% save_dir = "C:\Picture\数据采集-20250330\find_max_result_gama\";
% name = "19ms-14bit- 2025-03-30 10-40-54";

fid = fopen(input_dir + name + ".raw", 'r');
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
GrayImage = reshape(rawData,cols ,rows);
GrayImage = GrayImage - 16384;



fileID = fopen(save_dir + name + "-保存.raw", 'wb');
fwrite(fileID, GrayImage, 'uint16'); 
fclose(fileID);

min_val = find_min_nonzero(GrayImage(:));
max_val = max(GrayImage(:));
fprintf("max %d min %d \n" ,max_val ,min_val);

global_max = max(GrayImage(:));
[rows, cols] = size(GrayImage);
[row_index, col_index] = find(GrayImage == global_max);
fprintf("最大像素 %d \n" ,global_max);
fprintf("坐标 %d %d \n" ,row_index, col_index);

save_raw_image = uint8(round((GrayImage-min_val)/(max_val-min_val)*255));

outputPath = 'C:\Users\shike\Desktop\dde_14_result\a3.txt';
% 按行优先顺序保存为文本文件（每行一个值）
fileID = fopen(outputPath, 'w');
if fileID == -1
    error('无法创建或打开文件！');
end
matrix = save_raw_image;
% 按行优先遍历矩阵并写入文件
for row = 1:size(matrix, 1)
    for col = 1:size(matrix, 2)
        fprintf(fileID, '%04X\n', matrix(row, col));  % 保留6位小数
    end
end
fclose(fileID);
disp(['矩阵已按行优先顺序保存到: ', outputPath]);

save_temp_dir = strcat(save_dir ,name ,"8位图-线性拉伸.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(uint8(save_raw_image), save_temp_dir);

% 读取图像
img = save_raw_image; % 使用MATLAB自带的测试图像

img = rot90(img, -1);

% 设置对比度和亮度参数
contrast = 80;   % 对比度参数 (-100到100之间)
bright = 0;     % 亮度参数 (-255到255之间)

% 应用对比度增强
[enhancedImg1, enhancedImg2] = contrastEnhancement(img, contrast, bright);

% 显示结果
figure('Position', [100, 100, 1200, 300]);

subplot(1, 3, 1);
imshow(img);
title('原始图像');
axis on;

subplot(1, 3, 2);
imshow(enhancedImg1);
title('第一个增强函数');
axis on;

subplot(1, 3, 3);
imshow(enhancedImg2);
title('第二个增强函数');
axis on;

% 显示增强函数的效果曲线
figure('Position', [100, 100, 800, 500]);
x = linspace(0, 1, 100);
y1 = zeros(size(x));
y2 = zeros(size(x));
avg = mean(img(:));

for i = 1:length(x)
    y1(i) = avg + 100 * (x(i) - avg) / (100 - contrast) + bright/255;
    y2(i) = avg + (x(i) - avg) * (100 + contrast) / 100 + bright/255;
end

plot(x, x, 'k--', 'LineWidth', 1); % 原始映射线
hold on;
plot(x, y1, 'r', 'LineWidth', 2); % 第一个增强函数
plot(x, y2, 'b', 'LineWidth', 2); % 第二个增强函数
plot(avg, avg, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g'); % 平均灰度点

title('对比度增强函数映射曲线');
xlabel('原始灰度值');
ylabel('增强后灰度值');
legend('原始映射', '第一个函数', '第二个函数', '平均灰度');
grid on;

rgb_image = repmat(save_raw_image, [1, 1, 3]);
rgb_image = paint_image(rgb_image,row_index, col_index);
save_temp_dir = strcat(save_dir ,name ,"8位图-最大值位置.png");
save_temp_dir = char(save_temp_dir(1));
rgb_image = rot90(rgb_image, -1);
imwrite(uint8(rgb_image), save_temp_dir);


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


function [enhancedImg1, enhancedImg2] = contrastEnhancement(img, contrast, bright)
    % 对比度增强函数
    % 输入:
    %   img - 输入灰度图像
    %   contrast - 对比度参数 (-100到100之间)
    %   bright - 亮度参数
    % 输出:
    %   enhancedImg1 - 第一个增强函数处理后的图像
    %   enhancedImg2 - 第二个增强函数处理后的图像
    
    % 确保输入图像为double类型且在[0,1]范围内
    if ~isa(img, 'double')
        img = im2double(img);
    end
    
    % 计算平均灰度
    average_gray = mean(img(:));
    
    % 实现第一个对比度增强函数
    % Gray = Average_Gray + 100 * (Gray-Average_Gray)/(100-Contrast) + Bright
    enhancedImg1 = average_gray + 100 * (img - average_gray) / (100 - contrast) + bright/255;
    
    % 实现第二个对比度增强函数
    % Gray = Average_Gray + (Gray-Average_Gray) * (100 + Contrast)/100 + Bright
    enhancedImg2 = average_gray + (img - average_gray) * (100 + contrast) / 100 + bright/255;
    
    % 裁剪到[0,1]范围
    enhancedImg1 = max(min(enhancedImg1, 1), 0);
    enhancedImg2 = max(min(enhancedImg2, 1), 0);
end