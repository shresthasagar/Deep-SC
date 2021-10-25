function Y= ColumnPositive(X)
%COLUMNPOSITIVE Make each column of X positive as possible
%   Detailed explanation goes here

[m,n] = size(X);
Y = X;
for ii = 1:n
    if sum(X(:,ii))<0
    Y(:,ii) = -X(:,ii);
end

end

