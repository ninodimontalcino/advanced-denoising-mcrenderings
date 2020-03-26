function imageDenosing_simple(file_name, f, r, k) 
    %% Read input image and variance estimate
    dat = exrread(strcat(file_name, '.exr'));
    
    % Compute per channel variance
    [x,y,z] = size(dat);
    var_x = var(reshape(dat(:,:,1),[],1));
    var_y = var(reshape(dat(:,:,2),[],1));
    var_z = var(reshape(dat(:,:,3),[],1));
    var_max = max([var_x, var_y, var_z]);
    datvar = cat(3, repmat(var_x/var_max,[x,y]),repmat(var_y/var_max,[x,y]),repmat(var_z/var_max,[x,y]));
                
    flt = zeros(size(dat));
    wgtsum = zeros(size(dat));
    
    %% Loop over neighbors
    for dx = -r:r
        for dy= -r:r
           
            % Compute distance to neighbours
            ngb =  circshift(dat,[dx,dy]);
            
            % Uniform variance distance
            d2pixel = (((dat - ngb).^2) - 2*datvar) ...
               ./ ( eps + k^2 * 2*datvar);
            box_f = ones(2*f+1);
            
            % Box-Filtering
            d2patch = convn(d2pixel, box_f, 'same');
            wgt = exp(-max(0, d2patch));
            box_fm1 = ones(2*f-1);
            
            % Box filter weights for patch contribution
            wgt = convn(wgt, box_fm1, 'same');
            flt = flt + wgt .* ngb;
            wgtsum = wgtsum + wgt;
            
        end
    end
    
    %% Normalize and write denoise image to exr-file
    flt = flt./wgtsum;
    exrwrite(flt, strcat(file_name, '_denoised_simple_r=', int2str(r), '.exr'));
end