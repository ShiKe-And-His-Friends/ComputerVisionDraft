function dst = gaussianblur(image, r, sigma)
mask = zeros(2*r+1,2*r+1);
dst = image;
for i = 1:2*r+1
    for j = 1:2*r+1
        mask(i,j) = exp( -(power(i-r-1,2)+power(j-r-1,2)) / (2*sigma^2) );
    end
end
mask = mask/sum(mask(:));
[w, h]  = size(image);
image = double(image);
%mask
%image(1:2*r+1,1:2*r+1)
for i = 1:w
    for j = 1:h
        ys = i-r;
        ye = i+r;
        xs = j-r;
        xe = j+r;
        if xs<1
            xs = 1;
        end
        if xe>h
            xe = h;
        end
        if ys<1
            ys=1;
        end
        if ye>w
            ye=w;
        end


        dst(i,j) = sum(sum((image(ys:ye,xs:xe).*mask(ys-i+r+1:ye-i+r+1,xs-j+r+1:xe-j+r+1))));
    end
end
% img_o = CLHE(dst,1500);
% imshow(img_o,[0 1023]);
%return dst
end