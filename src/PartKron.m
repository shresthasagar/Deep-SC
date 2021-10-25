function Z = PartKron(Acell,Bcell)
%PARTKRON  Acell = {A_1,...,A_R}
%   Bcell = {B_1,...,B_R}
%  Acell o Bcell = [A_1 x B1,...., A_R x B_R]
    Z = [];
    for ii = 1:length(Acell)
        c = kron(Acell{ii},Bcell{ii});
        Z = [Z,c];
    end
end

