
function [q ,cof ,e] = GuideFilter_low(I, p, windows_size)
    
    %���ڰ뾶��С
    r = floor((windows_size -1)/2); %3 5 7 9
    disp("�����˲��뾶 = " + r);

    [hei, wid] = size(I);
    N = boxfilter(ones(hei, wid), r); 

    mean_I = boxfilter(I, r) ./ N;
    mean_p = boxfilter(p, r) ./ N;
    mean_Ip = boxfilter(I.*p, r) ./ N;
    % this is the covariance of (I, p) in each local patch.
    cov_Ip = mean_Ip - mean_I .* mean_p; 
    
    mean_II = boxfilter(I.*I, r) ./ N;
    var_I = mean_II - mean_I .* mean_I;
    
%     cov_Ip(cov_Ip < 1.0) = 0;
%     var_I(var_I < 1.0) = 0;
    
    %Э����
%     cov_Ip_max = max(max(cov_Ip));
%     cov_Ip_min = min(min(cov_Ip));
%     cov_Ip_mean = mean(mean(cov_Ip));
%     fprintf("Э���� max %f min %f mean %f \n",cov_Ip_max ,cov_Ip_min ,cov_Ip_mean);

    %����Ӧ����������
     %mean_sigma2_I = mean(cov_Ip(:));

    %����Ӧ���������� < threshold
%      threshold_cov_IP = cov_Ip(cov_Ip < 5000);
%      mean_sigma2_I = mean(threshold_cov_IP);

     %����Ӧ���������� < threshold
    %     threshold = 5000;
    %     n = 0 ;sum = 0;
    %     for i = 1:size(cov_Ip, 1) % ������
    %         for j = 1:size(cov_Ip, 2) % ������
    %             val = cov_Ip(i, j);
    %             if  val >= 1.0 && val<=threshold
    %                 n = n + 1.0;
    %                 sum = sum + val;
    %             end
    %         end
    %     end
    %     mean_sigma2_I = sum / n;

    %�������threshold��ͳ�Ƽ����ֵ
    max_sigma2_threhold = 1200000;
    n = 0;sum = 0;invalid_sum = 0;
    for i = 1:size(cov_Ip, 1) % ������
        for j = 1:size(cov_Ip, 2) % ������
             % С�����1λ
             val = round(cov_Ip(i, j)  * 10) / 10;
             cov_Ip(i, j) = val;
             
             % ����FPGAλ���ǿ�Ƹ�ֵ
            if cov_Ip(i, j) > max_sigma2_threhold
                cov_Ip(i, j) = max_sigma2_threhold * 0.4; % ϸ�ڲ���ʧ���Ŵ󲻹���
                invalid_sum = invalid_sum + 1.0;
            else 
            % С��FPGAλ��Ľ���ͳ�����Ƿֲ�
                n = n + 1.0;
                sum = sum + val;
            end
        end
    end
    fprintf("Э���� ������ֵ���� %d \n" ,invalid_sum);
    mean_sigma2_I = sum / n;
    disp("����Ӧֵepsilon = " + mean_sigma2_I);
  
    %TODO epsilonС��1
    %TODO ��ʽ��ĸ����
    
    %a = cov_Ip ./ (var_I + mean_sigma2_I);
    a = cov_Ip ./ (cov_Ip + 50);
    
    b = mean_p - a .* mean_I; 

    mean_a = boxfilter(a, r) ./ N;
    mean_b = boxfilter(b, r) ./ N;
    
    %�˲����ͼƬ
    q = mean_a .* I + mean_b;
    
    %�ݶ���ϢAk
    cof = mean_a;
    %����Ӧֵ
    e = mean_sigma2_I;
    
end

% �л����˲�
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