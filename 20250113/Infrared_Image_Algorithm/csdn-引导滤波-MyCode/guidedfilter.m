function [q ,cof ,e] = guidedfilter(I, p, r, eps)

%   - guidance image: I (should be a gray-scale/single channel image)
%   - filtering input image: p (should be a gray-scale/single channel image)
%   - local window radius: r
%   - regularization parameter: eps

    [hei, wid] = size(I);
    N = boxfilter(ones(hei, wid), r); 

    mean_I = boxfilter(I, r) ./ N;
    mean_p = boxfilter(p, r) ./ N;
    mean_Ip = boxfilter(I.*p, r) ./ N;
    % this is the covariance of (I, p) in each local patch.
    cov_Ip = mean_Ip - mean_I .* mean_p; 

    mean_II = boxfilter(I.*I, r) ./ N;
    var_I = mean_II - mean_I .* mean_I;

    % ·½²î
    mean_sigma2_I = mean(cov_Ip(:));
    disp("epsilon = " + mean_sigma2_I);

    a = cov_Ip ./ (var_I + mean_sigma2_I); 
    b = mean_p - a .* mean_I; 

    mean_a = boxfilter(a, r) ./ N;
    mean_b = boxfilter(b, r) ./ N;
    
    q = mean_a .* I + mean_b; 
    cof = mean_a;
    e = mean_sigma2_I;
end


function imDst = boxfilter(imSrc, r)

    %   BOXFILTER   O(1) time box filtering using cumulative sum
    %
    %   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    %   - Running time independent of r; 
    %   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
    %   - But much faster.

    [hei, wid] = size(imSrc);
    imDst = zeros(size(imSrc));

    %cumulative sum over Y axis
    imCum = cumsum(imSrc, 1);
    %difference over Y axis
    imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
    imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
    imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);

    %cumulative sum over X axis
    imCum = cumsum(imDst, 2);
    %difference over Y axis
    imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
    imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
    imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
end
