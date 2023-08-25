function [Y,D] = ColumnNormalization( X )
%UNTITLED normalize each column of X
%   To make its 2-norm to 1
[m,n]=size(X);
Y = zeros(m,n);
D = zeros(1,n);

for ii = 1:n
    D(ii) = norm(X(:,ii));
    if D(ii) ==0
        Y(:,ii) = X(:,ii);
    else
        Y(:,ii) = X(:,ii)/D(ii);
    end
     
end
    
end

