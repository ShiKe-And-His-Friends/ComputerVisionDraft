% ��ά����ķ�����Сֵ
function min_nonzero = find_min_nonzero(A)
    min_nonzero = Inf; % �Ƚ���Сֵ��ʼ��Ϊ�����
    [rows, cols] = size(A);
    for i = 1:rows
        for j = 1:cols
            if A(i, j) ~= 0 && A(i, j) < min_nonzero
                min_nonzero = A(i, j);
            end
        end
    end
    if min_nonzero == Inf
        min_nonzero = 0; % �������ȫΪ�㣬����0
    end
end