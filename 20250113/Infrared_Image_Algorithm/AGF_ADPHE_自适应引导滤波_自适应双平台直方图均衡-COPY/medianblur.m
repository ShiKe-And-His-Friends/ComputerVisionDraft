function dst = medianblur(src , r)
  dst = src;
  [w,h] = size(src);
  for i = r+1:w-r
    for j = r+1:h-r
      mask = src(i-r:i+r,j-r:j+r);
      dst(i,j) = median(median(mask));
    end
  end
end