% https://zhuanlan.zhihu.com/p/466219660

clear;
close all;

%文件
input_dir = "D:\Document\均值相差500+图像数据\test\dump\";
save_dir = "D:\Document\均值相差500+图像数据\test\Retinex\";
name = "场景4-AGF_ADPHE";

input_temp_dir = strcat(input_dir ,name ,"-分层基底图.raw");
input_temp_dir = char(input_temp_dir(1));

fid = fopen(input_temp_dir, 'r');
cols = 640;
rows = 512;
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
I = reshape(rawData,cols ,rows);


%分别取R G B 的三个分量，并将其转化为双精度
R = I(:,:,1);
R0 = double(R);

[N1 ,M1] = size(R);

%对R分量做对数变换
Rlog = log(R0 + 1);
%对R分量做二位傅里叶变换
Rfft2 = fft2(R0);
%形成高斯滤波函数(sigma = 128)
sigma = 128;
F = zeros(N1 ,M1);

for i = 1:N1
    for j = 1:M1
           F(i,j) = exp(-((i-N1/2)^2 +(j-M1/2)^2)/(2*sigma*sigma)); 
    end
end
F  = F./(sum(F(:)));

%对高斯滤波函数进行二维傅里叶变换
Ffft = fft2(double(F));

%对R分量和高斯滤波数进行卷积计算
DR0 = Rfft2 .* Ffft;
DR = ifft2(DR0);

%在对数域中，用原图像减去低通滤波图象，得到高频增强图象
DRdouble = double(DR);
DRlog = log(DRdouble+1);
Rr0 = Rlog - DRlog;


%形成高斯滤波函数（sigma = 256）
sigma = 256;
F = zeros(N1 ,M1);
for i = 1:N1
    for j = 1:M1
        F(i,j) = exp(-((i-N1/2)^2 +(j-M1/2)^2)/(2*sigma*sigma)); 
    end
end
F = F./(sum(F(:)));

%对高斯滤波函数进行二维傅里叶变换
Ffft = fft2(double(F));

%对R分量和高斯滤波数进行卷积计算
DR0 = Rfft2 .* Ffft;
DR = ifft2(DR0);

%在对数域中，用原图减去低通滤波后的图象，得到高频增强的图象
DRdouble = double(DR);
DRlog = log(DRdouble + 1);
Rr1 = Rlog - DRlog;


%形成高斯滤波函数（sigma = 512）
sigma = 512;
F = zeros(N1 ,M1);
for i = 1:N1
    for j = 1:M1
        F(i,j) = exp(-((i-N1/2)^2 +(j-M1/2)^2)/(2*sigma*sigma)); 
    end
end
F = F./(sum(F(:)));

%对高斯滤波函数进行二维傅里叶变换
Ffft = fft2(double(F));

%对R分量和高斯滤波数进行卷积计算
DR0 = Rfft2 .* Ffft;
DR = ifft2(DR0);

%在对数域中，用原图减去低通滤波后的图象，得到高频增强的图象
DRdoule = double(DR);
DRlog = log(DRdouble + 1);
Rr2 = Rlog - DRlog;

%对上述三次增强的图像取均值得到最终增强的图像
Rr = (1/3) * (Rr0 + Rr1 + Rr2);

%定义色彩回复算子C
a = 125;
% II = imadd(R0,R0);
% II = imadd(II,R0);
II = R0;
Ir = immultiply(R0 ,a);
C = imdivide(Ir , II);
C = log(C+1);

%将增强后的R分量乘以色彩恢复因子，并对其进行反对数变换
Rr = immultiply(C,Rr);
EXPRr = exp(Rr);

%对增强后的R分量进行灰度拉伸
%MIN = min(min(EXPRr));%非0
MIN = find_min_nonzero(EXPRr);
MAX = max(max(EXPRr));
EXPRr = (EXPRr-MIN)/(MAX-MIN);

image = EXPRr; % 假设这是要进行自适应直方图均衡化的图像
if any(isnan(image(:)))
    % 处理包含NaN的情况，可以选择将NaN替换为合理的值（比如图像均值、中位数等）
    mean_value = mean(image(~isnan(image))); % 计算非NaN像素的均值
    image(isnan(image)) = mean_value; % 将NaN替换为均值
end
EXPRr = image;

EXPRr = adapthisteq(EXPRr);


%对增强后的图像R G B 分量进行融合
I0 = uint8(round(EXPRr * 255));

subplot(121),imshow(I);
subplot(122),imshow(I0);

save_temp_dir = strcat(save_dir ,name ,"-融合前的Retinex基底图.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(I0, save_temp_dir);





