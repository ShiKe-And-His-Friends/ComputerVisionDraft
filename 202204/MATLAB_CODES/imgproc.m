function out = imgrayscaling(varargin)
% IMGRAYSCALING     执行灰度拉伸功能
%   语法：
%       out = imgrayscaling(I, [x1,x2], [y1,y2]);
%       out = imgrayscaling(X, map, [x1,x2], [y1,y2]);
%       out = imgrayscaling(RGB, [x1,x2], [y1,y2]);
%   这个函数提供灰度拉伸功能，输入图像应当是灰度图像，但如果提供的不是灰度
%   图像的话，函数会自动将图像转化为灰度形式。x1，x2，y1，y2应当使用双精度
%   类型存储，图像矩阵可以使用任何MATLAB支持的类型存储。

[A, map, x1 , x2, y1, y2] = parse_inputs(varargin{:});

% 计算输入图像A中数据类型对应的取值范围
range = getrangefromclass(A);
range = range(2);

% 如果输入图像不是灰度图，则需要执行转换
if ndims(A)==3,% A矩阵为3维，RGB图像
  A = rgb2gray(A);
elseif ~isempty(map),% MAP变量为非空，索引图像
  A = ind2gray(A,map);
end % 对灰度图像则不需要转换
 
% 读取原始图像的大小并初始化输出图像
[M,N] = size(A);
I = im2double(A);		% 将输入图像转换为双精度类型
out = zeros(M,N);
 
% 主体部分，双级嵌套循环和选择结构
for i=1:M
    for j=1:N
        if I(i,j)<x1
            out(i,j) = y1 * I(i,j) / x1;
        elseif I(i,j)>x2
            out(i,j) = (I(i,j)-x2)*(range-y2)/(range-x2) + y2;
        else
            out(i,j) = (I(i,j)-x1)*(y2-y1)/(x2-x1) + y1;
        end
    end
end

% 将输出图像的格式转化为与输入图像相同
if isa(A, 'uint8') % uint8
    out = im2uint8(out);
elseif isa(A, 'uint16')
    out = im2uint16(out);
% 其它情况，输出双精度类型的图像
end

 % 输出:
if nargout==0 % 如果没有提供参数接受返回值
  //imshow(out);
  return;
end