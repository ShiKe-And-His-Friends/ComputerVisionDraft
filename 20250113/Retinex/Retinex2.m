% https://zhuanlan.zhihu.com/p/391788215
clc
clear all;
I=imread('D:\Document\��ֵ���500+ͼ������\test\dump\����3-AGF_ADPHE-�ں�ǰ�Ļ���ͼ.png');
tic

R=I(:,:);
M=histeq(R);
In=M;

imshow(In);
figure
imshowpair(I, In, 'montage');
t=toc;

[C,L]=size(I); %��ͼ��Ĺ��
Img_size=C*L; %ͼ�����ص���ܸ���
G=256; %ͼ��ĻҶȼ�
H_x=0;
nk=zeros(G,1);%����һ��G��1�е�ȫ�����
for i=1:C
    for j=1:L
        Img_level=I(i,j)+1; %��ȡͼ��ĻҶȼ�
        nk(Img_level)=nk(Img_level)+1; %ͳ��ÿ���Ҷȼ����صĵ���
    end
end

clear all;
clc;
tic
%һ��ͼ���Ԥ���������ɫͼ����ҶȻ�
PS=imread('D:\Document\��ֵ���500+ͼ������\test\dump\����3-AGF_ADPHE-�ں�ǰ�Ļ���ͼ.png');                %����BMP��ɫͼ���ļ�
imshow(PS)                                  %��ʾ����
title('�����ͼ��')
%imwrite(rgb2gray(PS),'PicSampleGray.bmp'); %����ɫͼƬ�ҶȻ�������
R=PS(:,:,1);                          %�ҶȻ�������ݴ�������

%��������ֱ��ͼ
[m,n]=size(R);                             %����ͼ��ߴ����
GP=zeros(1,256);                            %Ԥ������ŻҶȳ��ָ��ʵ�����
for k=0:255
     GP(k+1)=length(find(R==k))/(m*n);      %����ÿ���Ҷȳ��ֵĸ��ʣ��������GP����Ӧλ��
end
figure,
bar(0:255,GP,'g')                    %����ֱ��ͼ
title('����ͼ���ֱ��ͼ')
xlabel('�Ҷ�ֵ')
ylabel('���ָ���')
%����ֱ��ͼ���⻯
S1=zeros(1,256);
for i=1:256
     for j=1:i
          S1(i)=GP(j)+S1(i);                 %����Sk
     end
end
S2=round((S1*256)+0.5);                          %��Sk�鵽������ĻҶ�
for i=1:256
     GPeq(i)=sum(GP(find(S2==i)));           %��������ÿ���Ҷȼ����ֵĸ���
end
figure,bar(0:255,GPeq,'b')                  %��ʾ���⻯���ֱ��ͼ
title('���⻯���ֱ��ͼ')
xlabel('�Ҷ�ֵ')
ylabel('���ָ���')
%�ģ�ͼ����⻯
PA=R;
for i=0:255
     PA(find(R==i))=S2(i+1);                %���������ع�һ����ĻҶ�ֵ�����������
end
imwrite(PA, 'D:\Document\��ֵ���500+ͼ������\test\dump\retinex3-1.png');

clear all;
tic
I = imread('D:\Document\��ֵ���500+ͼ������\test\dump\����3-AGF_ADPHE-�ں�ǰ�Ļ���ͼ.png');
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
imwrite(EXPRr, 'D:\Document\��ֵ���500+ͼ������\test\dump\retinex3-2.png');
imhist(EXPRr);
figure;