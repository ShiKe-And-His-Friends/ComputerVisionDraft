%% 距离和分水岭变换分割二值图像
f = imread('./data/Fig0219(a).tif');
imshow(f);

gc = ~f;
D = bwdist(gc);
L = watershed(-D);
w = L == 0;
g2 = f & ~w;
figure ,imshow(g2);

%%
% 使用梯度的分水岭分割
f = imread('./data/Fig0219(a).tif');
imshow(f);
h = fspecial('sobel');
fd = im2double(f);
g = sqrt(imfilter(fd ,h ,'replicate').^2 + imfilter(fd ,h','replicate').^2);
figure,imshow(g);
L = watershed(g);
wr = L == 0;
figure,imshow(wr);
