function [ cdf_matrix ] = make_cdf_matrix(p_image, block_size,limit)
%MAKE_CDF_MATRIX creates a CDF array for every tile in the input image

M = size(p_image,1);
N = size(p_image,2);
M_block = block_size(1);
N_block = block_size(2);
T1 = M/M_block;
T2 = N/N_block;
max_gray_value = max(p_image(:));
min_gray_value = min(p_image(:));
cdf_matrix = zeros(T1,T2,65534);

% iterate through the tiles
for i = 1:T1
    for j = 1:T2
        % extract a block
        a = (i-1)*M_block+1;
        b = i*M_block;
        c = (j-1)*N_block+1;
        d = j*N_block;
        current_block = p_image(a:b,c:d);
        current_block = uint16(current_block);
        % find normalized histogram
        [row, col] = size(current_block);
        frequency = zeros(1, 65534);
        cdf = zeros(1,65534);
 
        for m = 1:row
            for n = 1:col
                index = current_block(m, n);
                frequency(1,index) = frequency(1,index) + 1;
            end
        end
        gray_level = 0;
        for m = 1:65534
            if frequency(m)~=0
                gray_level = gray_level + 1;
            end
        end
        
        %找到直方图均衡的区间
        min_value = -1;
        max_value = -1;
        for m = 1:65534
            if frequency(m)<=0
                continue;
            end
            min_value = m;
            break;
        end
        m = 65534;
        while(max_value==-1)
            if frequency(m)<=0
                m = m -1;
                continue;
            end
            max_value = m;
        end 
        fprintf("gray_min = %d ,gary_max = %d ,gray_level = %d ,gray_grade = %d\n",min_value ,max_value,gray_level ,max_value-min_value+1);        
        
        % clip the normalized histogram 
        sum_val = 0;
        sum_count = 0;
        limit = floor(limit * row *col);
        %for m = 1:65534
        for m = min_value : max_value
%             if frequency(m)<=0
%                 continue;
%             end
           if (frequency(m)>limit) 
               sum_val = sum_val + (frequency(m)-limit);
               frequency(m) = limit;
           end
           sum_count = sum_count + 1;
        end
        
%         %平均分配
%         average_add = sum_val/sum_count;
%         %for m = 1:65534
%         for m = min_value : max_value
% %             if frequency(m)<=0
% %                 continue;
% %             end
%             frequency(m) = frequency(m) +average_add;
%         end  
        
        %除最重复值外平均分配
        while(sum_val > 0)
            for m = min_value : max_value
                if frequency(m)<=0
                    continue;
                end
                if frequency(m) < limit
                    frequency(m) = frequency(m) + 1;
                end
                sum_val = sum_val - 1;
                if (sum_val <= 0)
                    break;
                end
            end
        end

        sum_val = sum(frequency(1,:));
        frequency = double(frequency) ./ double(sum_val);
        
        cdf(1,1)=frequency(1,1);
        % HE直方图均衡化
        for m = 2:65534
            cdf(1,m) = cdf(1,m-1) + frequency(1,m);
        end
 
        if gray_level < 255
%             for m = 1:65534
%                 cdf2(1,m) = m;
%             end
            cdf2 = cdf * gray_level; %8bitwrong 16bitfine
        else
            cdf2 = cdf * 255;
        end
        
        % 第一公式
        %cdf2 = cdf * (max_value - min_value + 1) + min_value; %8bitwrong 16bitfine
        
        % 第二公式
        %cdf2 = cdf * 255; %fine
        
        % 第三公式
        %cdf2 = cdf * (max_gray_value - min_gray_value + 1)  + double(min_gray_value); %fine
        
        % 第四公式
        %cdf2 = cdf * (max_value - min_value + 1)  + double(min_gray_value);
        
        cdf_matrix(i,j,:) = cdf2;
          
    end
end

end

