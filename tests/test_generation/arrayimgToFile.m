function arrayimgToFile(filename, array)
    file = fopen(filename, 'w');
    for i = 1:600
        for j = 1:800
            fprintf(file, '%f %f %f ', array(i,j,1), array(i,j,2), array(i,j,3));
        end
        fprintf(file, '\n');
    end
    fclose(file)
end