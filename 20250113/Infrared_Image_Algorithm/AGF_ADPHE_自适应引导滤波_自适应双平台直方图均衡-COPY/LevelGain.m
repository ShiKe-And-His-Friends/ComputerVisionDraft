function newImage = LevelGain(GrayImage)

    %level = [0.01,0.03,0.06,0.11,0.20,0.4,0.6];
    gain_level = 0.01; %0.008; %增益挡位
    
    image = GrayImage;
    histogram = imhist(image);
    max_gray = max(histogram); 
    
    min_number = 1;
    while histogram(min_number) < max_gray* gain_level
        min_number = min_number + 1;
    end
    max_number = 256;
    while histogram(max_number) < max_gray* gain_level
        max_number = max_number - 1;
    end
    fprintf("挡位增益 最小 %d  ,最大 %d \n",min_number,max_number);
    
    image = double(image);
    newImage = floor(((255 - 0) * (image - min_number) / (max_number - min_number)) + 0);
    newImage = uint8(newImage);
    
end