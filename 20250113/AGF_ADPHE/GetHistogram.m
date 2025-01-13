% ͳ��ֱ��ͼ
function frequencyHist = GetHistogram(Image)

    [rows,cols] = size(Image);
    % compute the frequency of the image
    hist = zeros(1,16384); %14bitsλ��
    
    for i = 1:rows
        for j = 1:cols
            hist(Image(i,j)+1) = hist(Image(i,j)+1) + 1;
        end
    end
    
    frequencyHist = hist;
end