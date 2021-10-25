function Z= Cell2Mat(Acell)
%CELL2MAT Acell is a set of a number of matrices of the same size
%   
Z =[];
for ii = 1:length(Acell)
    Z = [Z,Acell{ii}];
end
end

