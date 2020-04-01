function rmse = calculateRMSE(filename1, filename2)

    y = exrread(strcat(filename1, '.exr'));
    yhat = exrread(strcat(filename2, '.exr'));

    rmse = sqrt(immse(y, yhat));
    
end