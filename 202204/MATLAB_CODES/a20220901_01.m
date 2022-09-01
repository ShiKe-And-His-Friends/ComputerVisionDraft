%% imshow 
clc 
clear

I =  imread('.\data\matlab_book_example_picture\dipum_images_ch02\Fig0203(a)(chest-xray).tif');
figure,subplot(121),imshow(I),subplot(122),imhist(I)
axis tight %% axis number a^10 values

%% dynamic range low
%% class(I)
%% min(I(:))
%% max(I(:))
figure,imshow(I,[])


