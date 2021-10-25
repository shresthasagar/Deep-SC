classdef helper
    methods (Static = true)
        
        function X = get_tensor(S, C)
            % Get the tensor outer product
            sizec = size(C);
            X = zeros(51,51,sizec(1));
            for rr=1:sizec(2)
                X = X + outprod(S(:,:,rr), C(:,rr));
            end
        end
        
        function A = pseudo_inverse(X)
            A = (inv(X'*X))*X';
        end

    end
end