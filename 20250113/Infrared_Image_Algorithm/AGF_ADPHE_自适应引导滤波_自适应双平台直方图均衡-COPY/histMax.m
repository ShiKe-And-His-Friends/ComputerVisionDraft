function y = histMax(src)
mmm = max(max(src))+1;
hist = zeros(1,mmm);
y = 0;
[w,h] = size(src);
for i = 1:w
    for j = 1:h
        hist(src(i,j)+1) =hist(src(i,j)+1)+ 1;
    end
end
m = hist(1);
s = 0;
for i = 1:mmm
    s = s + hist(i);
    if(hist(i)>m)
        m = hist(i);
        y = i-1;

    end
    if s*2>w*h
        break;
    end
end

end

