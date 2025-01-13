%边缘增强
%
function [gradient_magnitude] = EdgeEnhanceSobel(image)
    % 获取图像的高度和宽度
    [height, width] = size(image);
    % 定义Sobel算子的水平和垂直方向滤波核
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    % 初始化用于存储水平和垂直方向梯度分量以及梯度幅值的矩阵
    gradient_x = zeros(height, width);
    gradient_y = zeros(height, width);
    gradient_magnitude = zeros(height, width);
    % 循环遍历图像的每个像素（避开边界像素，边界像素可后续单独处理）
    for y = 2:height - 1
        for x = 2:width - 1
            % 提取当前像素周围3x3邻域的图像数据
            neighborhood = image(y - 1:y + 1, x - 1:x + 1);
            % 计算水平方向梯度分量
            gradient_x(y, x) = sum(sum(neighborhood.* sobel_x));
            % 计算垂直方向梯度分量
            gradient_y(y, x) = sum(sum(neighborhood.* sobel_y));
            % 计算梯度幅值（使用简化的平方和开方近似方式，更准确的可用sqrt函数）
            gradient_magnitude(y, x) = sqrt(gradient_x(y, x)^2 + gradient_y(y, x)^2);
        end
    end
    % 显示水平方向梯度分量图像（可按需查看垂直方向梯度分量及梯度幅值图像）
end