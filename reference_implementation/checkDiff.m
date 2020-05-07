function diff = checkDiff(a, b)

    y = exrread(a);
    yhat = exrread(b);
    
    diff = max(max(max(abs(y - yhat))));
    
end
    