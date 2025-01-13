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