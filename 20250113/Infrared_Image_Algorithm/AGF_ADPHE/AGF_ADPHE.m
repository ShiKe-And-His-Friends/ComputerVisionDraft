% 2024.12.24 sk95120
% 中波红外DDE：自适应引导滤波AGF 自适应双平台直方图均衡ADPHE
%
function AGF_ADPHE(GrayImage ,cols ,rows ,save_dir ,name)

    disp("###################  " + name + "  ################### " );
    
    %保存输入的图片数据
    fid = fopen(save_dir + name + "-输入DDE图像.raw", 'wb');
    fwrite(fid, GrayImage, 'uint16'); 
    fclose(fid);
    
    %配置原图和引导图
    input_mage = GrayImage;
    prompt_image = input_mage;
    
    %自适应引导滤波AGF
    windows_size = 3; %滤波窗口
    disp("引导滤波窗口大小 = " + windows_size);
    [result_image,coff_Ak_image ,epsilon] =AdaptiveGuideFilter(input_mage, prompt_image, windows_size);

    %基底base图和细节detail图
    base_image = result_image;
    base_image = round(base_image);
    base_image(base_image < 0) = 0;
    detail_image = input_mage - result_image;

    %图片像素的负值归零
    detail_image = round(detail_image);
    detail_image(detail_image < 0) = 0;

    %保存中间过程raw图
    fileID = fopen(save_dir + name + "-分层基底图.raw", 'wb');
    fwrite(fileID, base_image, 'uint16'); 
    fclose(fileID);
    fileID = fopen(save_dir + name + "-分层细节图.raw", 'wb');
    fwrite(fileID, detail_image, 'uint16'); 
    fclose(fileID);
    
    %TODO 图示：原始图的直方图统计

    %基底图做自适应双平台直方图均衡ADPHE
    frequencyHist = GetHistogram(base_image);
    
    %对概率密度函数做均值滤波
    
    %计算自适应双平台
    [pixel_mapping ,valid_num] = PixelMapping(frequencyHist ,size(base_image));

    %HE后的基底图
    base_adpt_image = uint8(zeros(rows,cols));
    max_val = max(max(base_image));
    min_val = min(min(base_image));
    fprintf("最大灰度 %d ,最小灰度 %d \n",max_val,min_val);
    for i=1:cols
         for j=1:rows
            idx = base_image(i,j);
            base_adpt_image(j,i) = pixel_mapping(idx);
         end
    end
    
    %基底图滤波1 用高斯滤波去噪
    %基底图滤波2 均值滤波器
    %基底图滤波3 引导滤波

    save_temp_dir = strcat(save_dir ,name ,"-融合前的基底图.png");
    save_temp_dir = char(save_temp_dir(1));
    imwrite(base_adpt_image, save_temp_dir);

    %细节图滤波-1 自定义滤波核
%     my_filter_core = 1 / 64 * [4 8 4; 8 16 8; 4 8 4];
%     filted_detail_image = imfilter(detail_image_temp,my_filter_core);
    
    %细节图滤波-2 Ak .* Detal
    %filted_detail_image = detail_image.* coff_Ak_image;
    %filted_detail_image(filted_detail_image < 0) = 0;
    
    %细节图3 高低增益
%     Gain_Max = 4.0;
%     Gain_Min = 1.5;
%     coff = Gain_Max* coff_Ak_image + Gain_Min;
%     filted_detail_image = detail_image.* coff;
%     filted_detail_image(filted_detail_image < 0) = 0;

    %细节图4 噪声掩模改进
    % 欧阳慧明,夏丽昆,李泽民,等.一种基于参数自适应引导滤波的红外图像细节增强算法[J].红外技术,2022,44(12):1324-1331.
    Gain_Alpha = 2;
    Gain_Max = 4.5;
    Gain_Min = 2.0;
    coff = Gain_Alpha * (Gain_Min +(Gain_Max -Gain_Min)*coff_Ak_image);
    filted_detail_image = detail_image.* coff;
    filted_detail_image(filted_detail_image < 0) = 0;
    
    %细节图滤波-3 用高斯滤波去噪
%     filter_size = [3 3];
%     sigma = 0.5;
%     gaussian_filter = fspecial('gaussian', filter_size, sigma);
%     filted_detail_image_temp = imfilter(filted_detail_image, gaussian_filter);
%     filted_detail_image = filted_detail_image_temp;

    fileID = fopen(save_dir + name + '-分层细节图-高斯滤波后.raw', 'wb');
    fwrite(fileID, filted_detail_image, 'uint16'); 
    fclose(fileID);

    %细节图转置
    filted_detail_image_T = double(zeros(rows,cols));
    for i=1:cols
         for j=1:rows
             filted_detail_image_T(j,i) = filted_detail_image(i,j);
        end
    end

    max_val = max(max(filted_detail_image_T));
    min_val = min(min(filted_detail_image_T));
    detail_adpt_image_copy = 255 * (filted_detail_image_T - min_val) / (max_val - min_val);%保存未压缩前的细节图
    
    %细节图1 线性拉伸
    %Gain_Thredhold_Detail = 0.7;
    %detail_adpt_image = Gain_Thredhold_Detail *255 * (filted_detail_image_T - min_val) / (max_val - min_val);%参与融合的压缩后的细节图
    
    %细节图2 线性填充
    %detail_adpt_image = min(filted_detail_image_T * 2,255.0); % log2/log14=0.2626 2/14 = 0.142 4
    
    %细节图3 高低增益
    detail_adpt_image = filted_detail_image_T;
    
    save_temp_dir = strcat(save_dir ,name ,"-融合前的细节图.png");
    save_temp_dir = char(save_temp_dir(1));
    imwrite(uint8(detail_adpt_image), save_temp_dir);
    
    %融合图片1 - detail * (2~10) + base * (0.6~1.5) 
    Gain_Thredhold_Base = 0.90; %0.9;
    fusion_image = uint8(min(Gain_Thredhold_Base * double(base_adpt_image) + 0.2 * detail_adpt_image, 255));

    %融合图片2 自适应融合 
    % [1]汪子君,罗渊贻,蒋尚志,等.基于引导滤波的自适应红外图像增强改进算法[J].光谱学与光谱分析,2020,40(11):3463-3467.
%     fusion_aplha = 0.12;
%     fusion_sigma = 0.1;
%     k_background = fusion_aplha + (valid_num / (65535 * 0.9))^fusion_sigma;
%     %k_detail = 1.08 - k_background;
%     k_detail = 1.0 - k_background;
%     disp("背景融合系数 = " + k_background + " 细节融合系数 = " + k_detail);
%     fusion_image = uint8(min(k_background * double(base_adpt_image) + k_detail * detail_adpt_image, 255));

    save_temp_dir = strcat(save_dir ,name ,"-DDE的融合图.png");
    save_temp_dir = char(save_temp_dir(1));
    imwrite(fusion_image, save_temp_dir);
    
    %融合图片3 等比例缩小
%     add_image = double(base_adpt_image) + double(filted_detail_image_T);
%     max_val = max(max(add_image));
%     min_val = min(min(add_image));
%     fusion_image = uint8(255 * (add_image - min_val) / (max_val - min_val));

    %挡位增益
%     level_gain_image = LevelGain(fusion_image);
%     fusion_image = level_gain_image;
    
    %评价图片质量
    EvaluateQuality(fusion_image);
    
end

function filtered_image = gaussian_filter_nonzero(image, sigma, window_size)
    % 获取图像的行数和列数
    [rows, cols] = size(image);
    % 初始化滤波后图像，与原图像大小相同
    filtered_image = zeros(rows, cols);
    % 计算窗口半径（假设窗口大小为奇数）
    half_window = floor(window_size / 2);
    
    % 生成二维高斯核
    [x, y] = meshgrid(-half_window:half_window, -half_window:half_window);
    gaussian_kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));
    
    % 遍历图像的每一个像素进行滤波操作
    for row = 1:rows
        for col = 1:cols
            % 仅对非零像素进行处理
            if image(row, col) ~= 0
                % 用于存储窗口内非零像素与高斯核乘积的和
                sum_value = 0;
                % 用于存储窗口内非零像素与高斯核乘积的和（用于归一化）
                sum_weight = 0;
                % 确定当前像素对应的滤波窗口范围
                row_start = max(1, row - half_window);
                row_end = min(rows, row + half_window);
                col_start = max(1, col - half_window);
                col_end = min(cols, col + half_window);
                % 遍历窗口内的像素
                for i = row_start:row_end
                    for j = col_start:col_end
                        if image(i, j) ~= 0
                            % 计算高斯核对应位置的值
                            kernel_value = gaussian_kernel(i - row_start + 1, j - col_start + 1);
                            sum_value = sum_value + image(i, j) * kernel_value;
                            sum_weight = sum_weight + kernel_value;
                        end
                    end
                end
                % 计算窗口内非零像素高斯滤波后的平均值，并赋值给滤波后图像
                if sum_weight > 0
                    filtered_image(row, col) = sum_value / sum_weight;
                end
            end
        end
    end
end
