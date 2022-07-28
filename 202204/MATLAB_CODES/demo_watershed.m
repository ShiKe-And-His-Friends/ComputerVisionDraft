%% 距离和分水岭变换分割二值图像
f = imread('./data/Fig0219(a).tif');
imshow(f);

gc = ~f;
D = bwdist(gc);
L = watershed(-D);
w = L == 0;
g2 = f & ~w;
figure ,imshow(g2);