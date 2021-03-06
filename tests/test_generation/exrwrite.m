function exrwrite( hdr, filename )
%EXRWRITE    Write OpenEXR high dynamic range (HDR) image.
% EXRWRITE(HDR, FILENAME) creates an OpenEXR high dynamic range (HDR) image
% file from HDR, a real numeric high dynamic range image with either one or
% three channels per pixel (i.e. hdr is a m-by-n or m-by-n-by-3 matrix).
% The HDR file with the name filename uses the PIZ wavelet based
% compression to minimize the file size.
%
% Note: this implementation uses the ILM IlmImf library version 1.6.1.
%
% See also EXRREAD,TONEMAP

    cv.imwrite(filename, hdr)
end
