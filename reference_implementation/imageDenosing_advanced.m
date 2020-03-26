function imageDenosing_advanced(file_name, f, r, k) 
    %% Read input image and variance estimate
    dat = exrread(strcat(file_name, '.exr'));
    datvar = exrread(strcat(file_name, '_variance.exr'));
    
    flt = zeros(size(dat));
    wgtsum = zeros(size(dat));
    
    %% Loop over neighbors
    for dx = -r:r
        for dy= -r:r
           
            % Shift the array to cover neighborhood
            ngb =  circshift(dat,[dx,dy]);
            ngbvar = circshift(datvar,[dx,dy]);
            
            % Non-uniform variance
            dist = (((dat - ngb).^2) - (datvar + min(ngbvar, datvar))) ...
               ./ ( eps + k^2 * (datvar + ngbvar));
            
            % Box-Filtering distance
            box_f = ones(2*f+1);
            d2patch = convn(dist, box_f, 'same');
            wgt = exp(-max(0, d2patch));
            
            % Box-Filter weights -> for patch contribution
            box_fm1 = ones(2*f-1);
            wgt = convn(wgt, box_fm1, 'same');
            flt = flt + wgt .* ngb;
            wgtsum = wgtsum + wgt;
            
        end
    end
    
    %% Normalize and write denoise image to exr-file
    flt = flt ./ wgtsum;
    exrwrite(flt, strcat(file_name, '_denoised_advanced_r=', int2str(r), '.exr'));
end