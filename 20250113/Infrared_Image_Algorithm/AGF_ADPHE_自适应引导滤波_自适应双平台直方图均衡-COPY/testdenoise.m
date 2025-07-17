%从bin文件读出的数据除以4
function [d, sigmaN] = testdenoise(data,pics)%单帧自己循环做100次则pics =0 ，外部循环每帧更新一次参数则pics=1
sigmaN = 9;
if pics == 0
    pics = 50;
else
    pics = 1;
end
for ps = 1:pics
    [d,sigmaN] = denoise(data,sigmaN,2,2,4,14); % image, sigmaN, r, sigma,ishift, databit
end

end