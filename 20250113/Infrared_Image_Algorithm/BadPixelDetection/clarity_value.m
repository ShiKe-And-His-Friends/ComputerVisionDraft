clc;
clear;

%��ȡ14bits��doubleС��rawͼ
cols = 640;
rows = 512;
 
input_dir = "D:\MATLAB_CODE\BadPixelDetection\2\";

%�����ļ����µ��ļ�
file_names = SearchDirectoryFiles(input_dir);

for i = 1:length(file_names)
    
    fid = fopen(input_dir + file_names(i) + ".bin", 'r');
    rawData = fread(fid, rows*cols, 'uint16');
    fclose(fid);
    GrayImage = reshape(rawData,cols ,rows);

    %оƬ��rawͼ��Ҫ�ü�
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
    %fprintf("������ % d\n",sum_val);
    fprintf("%d\n",sum_val);
    
end


%��ȡĿ¼���ļ��е��ļ���
function [fileNamesWithoutSuffix] = SearchDirectoryFiles(input_dir)

   % ָ���ļ�Ŀ¼�������Ե�ǰĿ¼Ϊ����������滻Ϊ����ľ���·�������·����
    directory = input_dir;
    % ��ȡָ��Ŀ¼�������ļ����ļ��е���Ϣ
    fileInfo = dir(directory);
    % ��ʼ��һ���յĵ�Ԫ�����飬���ڴ洢�ļ���
    fileNames = {};
    fileNamesWithoutSuffix = {};
    % ������ȡ������Ϣ��ɸѡ���ļ�������ӵ���Ԫ��������
    for i = 1:length(fileInfo)
        if fileInfo(i).name ~= "." && fileInfo(i).name ~= ".."
            % fileNames{end + 1} = fileInfo(i).name;
            % ͨ��strfind���������ļ��������һ��'.'��λ��
            dotIndex = strfind(fileInfo(i).name, '.');
            if ~isempty(dotIndex) % ȷ���ҵ���'.'�����к�׺�����
                % ��ȡ�ļ�������������׺����ȡ'.'֮ǰ�Ĳ���
                fileNameWithoutSuffix = fileInfo(i).name(1:dotIndex(end) - 1);
                % ���������ļ�����ӵ������Ԫ��������
                fileNamesWithoutSuffix{end + 1} = fileNameWithoutSuffix;
            else % ���û���ҵ�'.'��˵�����ܱ����û�к�׺��ֱ��ʹ��ԭ�ļ���
                fileNamesWithoutSuffix{end + 1} = fileInfo(i).name;
            end
        end
    end
    % ��ʾ�ļ����б��洢�ڵ�Ԫ�������У�
    disp(fileNamesWithoutSuffix);
end