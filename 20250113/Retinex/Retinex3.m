% https://zhuanlan.zhihu.com/p/466219660

clear;
close all;

%�ļ�
input_dir = "D:\Document\��ֵ���500+ͼ������\test\dump\";
save_dir = "D:\Document\��ֵ���500+ͼ������\test\Retinex\";
name = "����4-AGF_ADPHE";

input_temp_dir = strcat(input_dir ,name ,"-�ֲ����ͼ.raw");
input_temp_dir = char(input_temp_dir(1));

fid = fopen(input_temp_dir, 'r');
cols = 640;
rows = 512;
rawData = fread(fid, rows*cols, 'uint16');
fclose(fid);
I = reshape(rawData,cols ,rows);


%�ֱ�ȡR G B ������������������ת��Ϊ˫����
R = I(:,:,1);
R0 = double(R);

[N1 ,M1] = size(R);

%��R�����������任
Rlog = log(R0 + 1);
%��R��������λ����Ҷ�任
Rfft2 = fft2(R0);
%�γɸ�˹�˲�����(sigma = 128)
sigma = 128;
F = zeros(N1 ,M1);

for i = 1:N1
    for j = 1:M1
           F(i,j) = exp(-((i-N1/2)^2 +(j-M1/2)^2)/(2*sigma*sigma)); 
    end
end
F  = F./(sum(F(:)));

%�Ը�˹�˲��������ж�ά����Ҷ�任
Ffft = fft2(double(F));

%��R�����͸�˹�˲������о������
DR0 = Rfft2 .* Ffft;
DR = ifft2(DR0);

%�ڶ������У���ԭͼ���ȥ��ͨ�˲�ͼ�󣬵õ���Ƶ��ǿͼ��
DRdouble = double(DR);
DRlog = log(DRdouble+1);
Rr0 = Rlog - DRlog;


%�γɸ�˹�˲�������sigma = 256��
sigma = 256;
F = zeros(N1 ,M1);
for i = 1:N1
    for j = 1:M1
        F(i,j) = exp(-((i-N1/2)^2 +(j-M1/2)^2)/(2*sigma*sigma)); 
    end
end
F = F./(sum(F(:)));

%�Ը�˹�˲��������ж�ά����Ҷ�任
Ffft = fft2(double(F));

%��R�����͸�˹�˲������о������
DR0 = Rfft2 .* Ffft;
DR = ifft2(DR0);

%�ڶ������У���ԭͼ��ȥ��ͨ�˲����ͼ�󣬵õ���Ƶ��ǿ��ͼ��
DRdouble = double(DR);
DRlog = log(DRdouble + 1);
Rr1 = Rlog - DRlog;


%�γɸ�˹�˲�������sigma = 512��
sigma = 512;
F = zeros(N1 ,M1);
for i = 1:N1
    for j = 1:M1
        F(i,j) = exp(-((i-N1/2)^2 +(j-M1/2)^2)/(2*sigma*sigma)); 
    end
end
F = F./(sum(F(:)));

%�Ը�˹�˲��������ж�ά����Ҷ�任
Ffft = fft2(double(F));

%��R�����͸�˹�˲������о������
DR0 = Rfft2 .* Ffft;
DR = ifft2(DR0);

%�ڶ������У���ԭͼ��ȥ��ͨ�˲����ͼ�󣬵õ���Ƶ��ǿ��ͼ��
DRdoule = double(DR);
DRlog = log(DRdouble + 1);
Rr2 = Rlog - DRlog;

%������������ǿ��ͼ��ȡ��ֵ�õ�������ǿ��ͼ��
Rr = (1/3) * (Rr0 + Rr1 + Rr2);

%����ɫ�ʻظ�����C
a = 125;
% II = imadd(R0,R0);
% II = imadd(II,R0);
II = R0;
Ir = immultiply(R0 ,a);
C = imdivide(Ir , II);
C = log(C+1);

%����ǿ���R��������ɫ�ʻָ����ӣ���������з������任
Rr = immultiply(C,Rr);
EXPRr = exp(Rr);

%����ǿ���R�������лҶ�����
%MIN = min(min(EXPRr));%��0
MIN = find_min_nonzero(EXPRr);
MAX = max(max(EXPRr));
EXPRr = (EXPRr-MIN)/(MAX-MIN);

image = EXPRr; % ��������Ҫ��������Ӧֱ��ͼ���⻯��ͼ��
if any(isnan(image(:)))
    % �������NaN�����������ѡ��NaN�滻Ϊ�����ֵ������ͼ���ֵ����λ���ȣ�
    mean_value = mean(image(~isnan(image))); % �����NaN���صľ�ֵ
    image(isnan(image)) = mean_value; % ��NaN�滻Ϊ��ֵ
end
EXPRr = image;

EXPRr = adapthisteq(EXPRr);


%����ǿ���ͼ��R G B ���������ں�
I0 = uint8(round(EXPRr * 255));

subplot(121),imshow(I);
subplot(122),imshow(I0);

save_temp_dir = strcat(save_dir ,name ,"-�ں�ǰ��Retinex����ͼ.png");
save_temp_dir = char(save_temp_dir(1));
imwrite(I0, save_temp_dir);





