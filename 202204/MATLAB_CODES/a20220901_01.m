%% imshow 
clc 
clear

I =  imread('.\data\matlab_book_example_picture\dipum_images_ch02\Fig0203(a)(chest-xray).tif');
figure,subplot(121),imshow(I),subplot(122),imhist(I)
axis tight %% axis number a^10 values

%% dynamic range low
class(I)
min(I(:))
max(I(:))
figure,imshow(I,[])

%% save picture
clc,clear
f = imread('.\data\matlab_book_example_picture\dipum_images_ch02\Fig0206(a)(rose-original).tif');
figure,imshow(f);
print -f3 -dtiff -r300 .\data\process_on_way\hi_res_rose

%% im2uint8
clc
clear
f1 = [-0.5 ,0.5 ,0.75 ,1.5]
g1 = im2uint8(f1) ,f1
f2 = uint8(f1*100) ,f2
f3 = f1*100
g3 = im2uint8(f3) ,g3
f4 = uint16(f1 * 50000),f4
g4 = im2uint8(f4)

%% im2bw 
%% 0-1 white-black
clc,clear
I = imread('./data/Fig0228(a).tif');
imshow(I)
BW = im2bw(I ,0.36);
figure,imshow(BW)

%% imabsdiff
%%
clc,clear
I = imread('./data/Fig0217(a).tif');
imshow(I,[])
J3 = uint8(filter2(fspecial('gaussian') ,I));
imshow(J3,[]);






