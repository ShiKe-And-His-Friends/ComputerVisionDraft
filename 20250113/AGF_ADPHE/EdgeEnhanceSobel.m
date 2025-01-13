%��Ե��ǿ
%
function [gradient_magnitude] = EdgeEnhanceSobel(image)
    % ��ȡͼ��ĸ߶ȺͿ��
    [height, width] = size(image);
    % ����Sobel���ӵ�ˮƽ�ʹ�ֱ�����˲���
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    % ��ʼ�����ڴ洢ˮƽ�ʹ�ֱ�����ݶȷ����Լ��ݶȷ�ֵ�ľ���
    gradient_x = zeros(height, width);
    gradient_y = zeros(height, width);
    gradient_magnitude = zeros(height, width);
    % ѭ������ͼ���ÿ�����أ��ܿ��߽����أ��߽����ؿɺ�����������
    for y = 2:height - 1
        for x = 2:width - 1
            % ��ȡ��ǰ������Χ3x3�����ͼ������
            neighborhood = image(y - 1:y + 1, x - 1:x + 1);
            % ����ˮƽ�����ݶȷ���
            gradient_x(y, x) = sum(sum(neighborhood.* sobel_x));
            % ���㴹ֱ�����ݶȷ���
            gradient_y(y, x) = sum(sum(neighborhood.* sobel_y));
            % �����ݶȷ�ֵ��ʹ�ü򻯵�ƽ���Ϳ������Ʒ�ʽ����׼ȷ�Ŀ���sqrt������
            gradient_magnitude(y, x) = sqrt(gradient_x(y, x)^2 + gradient_y(y, x)^2);
        end
    end
    % ��ʾˮƽ�����ݶȷ���ͼ�񣨿ɰ���鿴��ֱ�����ݶȷ������ݶȷ�ֵͼ��
end