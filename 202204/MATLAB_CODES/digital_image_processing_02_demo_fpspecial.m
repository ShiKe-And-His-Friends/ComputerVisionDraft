%%
% fspecial 
% 生成拉普拉斯算子

w = fspecial('laplacian' ,0);
f = imread('.\\data\\Fig0217(a).tif');
imshow(f);
g1 = imfilter(f ,w ,'replicate');
figure,imshow(g1 ,[]);

f2 = tofloat(f);
g2 = imfilter(f2 ,w ,'replicate');
g = f2 - g2;
figure,imshow(g);

%%