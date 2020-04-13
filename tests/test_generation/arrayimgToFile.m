% Merge = 0: no merge. 1: simple merge. 2: variance merge
function arrayimgToFile(filename, array, merge)
    file = fopen(filename, 'w');
    for i = 1:600
        for j = 1:800
            if merge == 0
                fprintf(file, '%f %f %f ', array(i,j,1), array(i,j,2), array(i,j,3));
            elseif merge == 1
                fprintf(file, '%f ', (array(i,j,1) + array(i,j,2) + array(i,j,3))/3.0);
            else
                fprintf(file, '%f ', (array(i,j,1) + array(i,j,2) + array(i,j,3))/9.0);
            end
        end
        fprintf(file, '\n');
    end
    fclose(file)
end