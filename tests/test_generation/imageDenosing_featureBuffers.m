function imageDenosing_featureBuffers(file_name, f, r, k_c, k_f, t) 
    %% Read input
    
    % Read input image --> to be denoised
    dat = exrread(strcat(file_name, '.exr'));
    arrayimgToFile(strcat(file_name, '.txt'), dat, 0);
    
    % Read input features
    dat_var_color = exrread(strcat(file_name, '_variance.exr'));
    arrayimgToFile(strcat(file_name, '_variance.txt'), dat_var_color, 0);
    dat_var_normal = exrread(strcat(file_name, '_normal_variance.exr'));
    arrayimgToFile(strcat(file_name, '_normal_variance.txt'), dat_var_normal, 2);
    dat_var_albedo = exrread(strcat(file_name, '_albedo_variance.exr'));
    arrayimgToFile(strcat(file_name, '_albedo_variance.txt'), dat_var_albedo, 2);
    dat_var_depth = exrread(strcat(file_name, '_depth_variance.exr'));
    arrayimgToFile(strcat(file_name, '_depth_variance.txt'), dat_var_depth, 2);
    dat_normal = exrread(strcat(file_name, '_normal.exr'));
    arrayimgToFile(strcat(file_name, '_normal.txt'), dat_normal, 1);
    dat_depth = exrread(strcat(file_name, '_depth.exr'));
    arrayimgToFile(strcat(file_name, '_depth.txt'), dat_depth, 1);
    dat_albedo = exrread(strcat(file_name, '_albedo.exr'));
    arrayimgToFile(strcat(file_name, '_albedo.txt'), dat_albedo, 1);
    
    % Calculate feeature gradients
    sqrd_grad_normal = sqrd_gradient(dat_normal);
    sqrd_grad_depth = sqrd_gradient(dat_depth);
    sqrd_grad_albedo = sqrd_gradient(dat_albedo);
        
    % Init containers with zeros
    flt = zeros(size(dat));
    wgtsum = zeros(size(dat));
    
    %% Loop over neighbors
    for dx = -r:r
        for dy= -r:r
           
            % Shift the array to cover neighborhood
            ngb =  circshift(dat,[dx,dy]);
            ngb_var_color = circshift(dat_var_color,[dx,dy]);
            ngb_var_normal = circshift(dat_var_normal,[dx,dy]);
            ngb_var_albedo = circshift(dat_var_albedo,[dx,dy]);
            ngb_var_depth = circshift(dat_var_depth,[dx,dy]);
            ngb_normal = circshift(dat_normal,[dx,dy]);
            ngb_depth = circshift(dat_depth,[dx,dy]);
            ngb_albedo = circshift(dat_albedo,[dx,dy]);
            
            % Compute distance (color)
            dist_color = (((dat - ngb).^2) - (dat_var_color + min(ngb_var_color, dat_var_color))) ...
               ./ ( eps + k_c^2 * (dat_var_color + ngb_var_color));
            
            % Box-Filtering distance (color)
            box_f = ones(2*f+1);
            dist_color = convn(dist_color, box_f, 'same');
            wgt_color = exp(-max(0, dist_color));
            
            % Compute distance (assume noise-free features)
            % --> Pixel based distance => no box-filtering necessary
            dist_normal = ((dat_normal - ngb_normal).^2 - (dat_var_normal + min(ngb_var_normal, dat_var_normal)))....
                ./ (k_f^2 * max(t, max(dat_var_normal, sqrd_grad_normal)) ) ;
            
            dist_albedo = ((dat_albedo - ngb_albedo).^2 - (dat_var_albedo + min(ngb_var_albedo, dat_var_albedo))) ...
                ./ (k_f^2 * max(t, max(dat_var_albedo, sqrd_grad_albedo)) );
                   
             dist_depth = ((dat_depth - ngb_depth).^2 - (dat_var_depth + min(ngb_var_depth, dat_var_depth))) ...
                ./ (k_f^2 * max(t, max(dat_var_depth, sqrd_grad_depth)) );
            
            dist_features = max(max(dist_normal, dist_depth), dist_albedo);
            wgt_features = exp(-dist_features);
            
            
            % Compute joint weight
            wgt = min(wgt_color, wgt_features);
            %wgt = wgt_features;
            
            % Box-Filter weights 
            box_fm1 = ones(2*f-1);
            wgt = convn(wgt, box_fm1, 'same');
            flt = flt + wgt .* ngb;
            wgtsum = wgtsum + wgt;
            
        end
    end
    
    %% Normalize and write denoise image to exr-file
    flt = flt ./ wgtsum;
    arrayimgToFile(strcat(file_name, '_output.txt'), flt, 0);
    exrwrite(flt, strcat(file_name, '_denoised_fb_r=', int2str(r), '.exr'));
end


% Compute squard gradient --> following recipe from slides
function sqrd_grad = sqrd_gradient(features)

    gL = (features - circshift(features,[ 0, -1]))/2;
    gR = (features - circshift(features,[ 0,  1]))/2;
    gU = (features - circshift(features,[ 1,  0]))/2;
    gD = (features - circshift(features,[-1,  0]))/2;
    
    sqrd_grad = min(gL.^2, gR.^2) +  min(gU.^2, gD.^2);  
end


