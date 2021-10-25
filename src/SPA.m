function K = SPA(X,r)
    % SPA algorithm implementation
    % returns the indices of the endmembers
    R = X; 
    normX = sum(X.^2); 
    normR = normX; 
    
    K = zeros(1,r);
    
    i = 1; 
    % Perform r steps (unless the residual is zero)
    while i <= r && max(normR) > 1e-12 
        
        % Select the column of R with the greatest 2-norm
        [~ , max_index] = max(sum(R.^2)); 
        K(i) = max_index;
        
        % unit vector u for the bth colum
        u = R(:,K(i))/norm(R(:,K(i))); 
        R = R - u*(u'*R);
        
        normR = sum(R.^2); 
        i = i + 1; 
    end
    K = K(1, 1:i-1);
end 