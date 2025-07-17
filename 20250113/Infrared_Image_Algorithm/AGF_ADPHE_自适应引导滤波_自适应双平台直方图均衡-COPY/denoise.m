function [dst,sigmaN] = denoise( image, sigmaN ,r, sigma, ishift , databits )
    
  image = int64( image);%*pow2(ishift);
  gauss =int64( gaussianblur(image,r,sigma));
  res = int64(image - gauss );
  sigma1 = abs(res);
  sigma2 = int64(gaussianblur(sigma1,r,sigma));
  sigma3 = int64(power(sigma2,2));%/pow2(2*(ishift+databits)-31));
  sigmaU1 = int64(sigma3 - sigmaN);
  sigmaU2 = (abs(sigmaU1)+sigmaU1)/2;
  sigmaU3 = (sigmaU2./sigma3).*res;
%   sigmaU4 = gaussianblur(sigmaU3,r,sigma);
%   sigmaU5 = sigmaU4.*double(res);
  dst = gauss+int64(sigmaU3);
  dst = int64(dst);%int64(dst/pow2(ishift));

  %med = medianblur(sigma3, r);
  sigmaN = fix((9*double(sigmaN)+double(2*histMax(sigma3)))/10.0);

end



