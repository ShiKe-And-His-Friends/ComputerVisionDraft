f = imread('./data/Fig0219(a).tif');
count = 0;
T = mean2(f);
T
done = false;
while ~done 
    count = count + 1;
    g = f > T;
    Tnext = 0.5 * (mean(f(g)) + mean(f(~g)));
    done = abs(T - Tnext) < 0.5;
    T = Tnext;
end
g = im2bw(f ,T/255);
figure,imhist(f);
figure,imshow(g);

