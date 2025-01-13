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