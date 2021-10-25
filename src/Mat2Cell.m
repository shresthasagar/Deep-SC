function Mc = Mat2Cell(M,L)
%MAT2CELL Summary of this function goes here
%   Detailed explanation goes here
Mc = {};
for ii = 1:length(L)
    if ii==1
        Mc{ii} = M(:,1:L(ii));
    else
        Mc{ii} = M(:,sum(L(1:ii-1))+1:sum(L(1:ii)));
    end
end

