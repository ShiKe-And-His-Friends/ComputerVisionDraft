% https://zhuanlan.zhihu.com/p/391788215
clc
clear all;
I=imread('D:\Document\均值相差500+图像数据\test\dump\场景3-AGF_ADPHE-融合前的基底图.png');
tic

R=I(:,:);
M=histeq(R);
In=M;

imshow(In);
figure
imshowpair(I, In, 'montage');
t=toc;

[C,L]=size(I); %求图像的规格
Img_size=C*L; %图像像素点的总个数
G=256; %图像的灰度级
H_x=0;
nk=zeros(G,1);%产生一个G行1列的全零矩阵
for i=1:C
    for j=1:L
        Img_level=I(i,j)+1; %获取图像的灰度级
        nk(Img_level)=nk(Img_level)+1; %统计每个灰度级像素的点数
    end
end

clear all;
clc;
tic
%一，图像的预处理，读入彩色图像将其灰度化
PS=imread('D:\Document\均值相差500+图像数据\test\dump\场景3-AGF_ADPHE-融合前的基底图.png');                %读入BMP彩色图像文件
imshow(PS)                                  %显示出来
title('输入的图像')
%imwrite(rgb2gray(PS),'PicSampleGray.bmp'); %将彩色图片灰度化并保存
R=PS(:,:,1);                          %灰度化后的数据存入数组

%二，绘制直方图
[m,n]=size(R);                             %测量图像尺寸参数
GP=zeros(1,256);                            %预创建存放灰度出现概率的向量
for k=0:255
     GP(k+1)=length(find(R==k))/(m*n);      %计算每级灰度出现的概率，将其存入GP中相应位置
end
figure,
bar(0:255,GP,'g')                    %绘制直方图
title('雾天图像的直方图')
xlabel('灰度值')
ylabel('出现概率')
%三，直方图均衡化
S1=zeros(1,256);
for i=1:256
     for j=1:i
          S1(i)=GP(j)+S1(i);                 %计算Sk
     end
end
S2=round((S1*256)+0.5);                          %将Sk归到相近级的灰度
for i=1:256
     GPeq(i)=sum(GP(find(S2==i)));           %计算现有每个灰度级出现的概率
end
figure,bar(0:255,GPeq,'b')                  %显示均衡化后的直方图
title('均衡化后的直方图')
xlabel('灰度值')
ylabel('出现概率')
%四，图像均衡化
PA=R;
for i=0:255
     PA(find(R==i))=S2(i+1);                %将各个像素归一化后的灰度值赋给这个像素
end
imwrite(PA, 'D:\Document\均值相差500+图像数据\test\dump\retinex3-1.png');

clear all;
tic
I = imread('D:\Document\均值相差500+图像数据\test\dump\场景3-AGF_ADPHE-融合前的基底图.png');
R = I(:, :);
 %imhist(R);
% figure;
[N1, M1] = size(R);
R0 = double(R);
Rlog = log(R0+1);
Rfft2 = fft2(R0);
 
sigma = 200;
F = fspecial('gaussian', [N1,M1], sigma); 
Efft = fft2(double(F));
 
DR0 = Rfft2.* Efft;
DR = ifft2(DR0);
 
DRlog = log(DR +1);
Rr = Rlog - DRlog;
EXPRr = exp(Rr);
%imhist(EXPRr);
%figure;
MIN = min(min(EXPRr));
MAX = max(max(EXPRr));
EXPRr = (EXPRr - MIN)/(MAX - MIN)*255;
EXPRr=uint8(EXPRr);
imwrite(EXPRr, 'D:\Document\均值相差500+图像数据\test\dump\retinex3-2.png');
imhist(EXPRr);
figure;