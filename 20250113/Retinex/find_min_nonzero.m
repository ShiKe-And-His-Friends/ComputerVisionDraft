% 二维数组的非零最小值
function min_nonzero = find_min_nonzero(A)
    min_nonzero = Inf; % 先将最小值初始化为无穷大
    [rows, cols] = size(A);
    for i = 1:rows
        for j = 1:cols
            if A(i, j) ~= 0 && A(i, j) < min_nonzero
                min_nonzero = A(i, j);
            end
        end
    end
    if min_nonzero == Inf
        min_nonzero = 0; % 如果数组全为零，返回0
    end
end