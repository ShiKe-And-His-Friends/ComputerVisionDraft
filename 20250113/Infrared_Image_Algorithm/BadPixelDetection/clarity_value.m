clc;
clear;

%读取14bits的double小端raw图
cols = 640;
rows = 512;
 
input_dir = "D:\MATLAB_CODE\BadPixelDetection\2\";

%搜索文件夹下的文件
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".bin", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);

    %芯片的raw图需要裁剪
    GrayImage = GrayImage - 16384;
    
    %fprintf("%s \n",char(file_names(i)));
    
    start_row = floor((640 - 512) / 2) + 1;
    end_row = start_row + 512 - 1;
    sum_val = 0;
    for m = 2:rows-2
        for n = start_row:end_row
            neighborhood = GrayImage( n-1:n+1 ,m-1:m+1);
            center = neighborhood(2,2);
            p1 = abs(neighborhood(1,1)-center);
            p2 = abs(neighborhood(1,2)-center);
            p3 = abs(neighborhood(1,3)-center);
            p4 = abs(neighborhood(2,1)-center);
            p5 = abs(neighborhood(2,3)-center);
            p6 = abs(neighborhood(3,1)-center);
            p7 = abs(neighborhood(3,2)-center);
            p8 = abs(neighborhood(3,3)-center);
            sum_val = sum_val + (p1+p2+p3+p4+p5+p6+p7+p8);
        end
    end
    %fprintf("清晰度 % d\n",sum_val);
    fprintf("%d\n",sum_val);
    
end


%读取目录下文件夹的文件名
function [fileNamesWithoutSuffix] = SearchDirectoryFiles(input_dir)

   % 指定文件目录（这里以当前目录为例，你可以替换为具体的绝对路径或相对路径）
    directory = input_dir;
    % 获取指定目录下所有文件和文件夹的信息
    fileInfo = dir(directory);
    % 初始化一个空的单元格数组，用于存储文件名
    fileNames = {};
    fileNamesWithoutSuffix = {};
    % 遍历获取到的信息，筛选出文件名并添加到单元格数组中
    for i = 1:length(fileInfo)
        if fileInfo(i).name ~= "." && fileInfo(i).name ~= ".."
            % fileNames{end + 1} = fileInfo(i).name;
            % 通过strfind函数查找文件名中最后一个'.'的位置
            dotIndex = strfind(fileInfo(i).name, '.');
            if ~isempty(dotIndex) % 确保找到了'.'，即有后缀的情况
                % 提取文件名（不包含后缀），取'.'之前的部分
                fileNameWithoutSuffix = fileInfo(i).name(1:dotIndex(end) - 1);
                % 将处理后的文件名添加到结果单元格数组中
                fileNamesWithoutSuffix{end + 1} = fileNameWithoutSuffix;
            else % 如果没有找到'.'，说明可能本身就没有后缀，直接使用原文件名
                fileNamesWithoutSuffix{end + 1} = fileInfo(i).name;
            end
        end
    end
    % 显示文件名列表（存储在单元格数组中）
    disp(fileNamesWithoutSuffix);
end