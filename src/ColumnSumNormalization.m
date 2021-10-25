function [Y,D] = ColumnSumNormalization( X )
    % Normalize each column of X
    % To make its l1-norm sum to 1
    [m,n]=size(X);
    Y = zeros(m,n);
    D = zeros(1,n);
    
    for ii = 1:n
        D(ii) = sum(X(:,ii));
        if D(ii) == 0
            Y(:,ii) = X(:,ii);
        else
            Y(:,ii) = X(:,ii)/D(ii);
        end
         
    end
        
end
    
    