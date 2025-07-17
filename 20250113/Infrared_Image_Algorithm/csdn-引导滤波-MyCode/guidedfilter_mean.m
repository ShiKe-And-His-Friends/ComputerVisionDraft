function [q ,cof ,e] = guidedfilter_mean(I, p, r, eps)

%   - guidance image: I (should be a gray-scale/single channel image)
%   - filtering input image: p (should be a gray-scale/single channel image)
%   - local window radius: r
%   - regularization parameter: eps

    [hei, wid] = size(I);
    N = meanfilter(ones(hei, wid), r); 

    mean_I = meanfilter(I, r) ./ N;
    mean_p = meanfilter(p, r) ./ N;
    mean_Ip = meanfilter(I.*p, r) ./ N;
    % this is the covariance of (I, p) in each local patch.
    cov_Ip = mean_Ip - mean_I .* mean_p; 

    mean_II = meanfilter(I.*I, r) ./ N;
    var_I = mean_II - mean_I .* mean_I;
    
    a = cov_Ip ./ (var_I + eps); 
    b = mean_p - a .* mean_I; 

    mean_a = meanfilter(a, r) ./ N;
    mean_b = meanfilter(b, r) ./ N;

    mean_a_4096 = mean_a * 4096;
    mean_a_4096_max =  max(max(mean_a_4096));
    mean_a_4096_min =  min(min(mean_a_4096));
    disp("max = " + mean_a_4096_max);
    disp("min = " + mean_a_4096_min);
    
    q = mean_a .* I + mean_b; 
    cof = mean_a;
end


function imDst = meanfilter(imSrc, r)

    % 创建3x3的均值滤波器
    mean_filter = ones(3, 3) / 9;

    % 使用imfilter函数进行均值滤波
    imDst = imfilter(imSrc, mean_filter);
    
end
