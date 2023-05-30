classdef metric
    methods (Static = true)
        
        function error = SRE(T_true, T_estimated)
            error = (frob(T_true - T_estimated)^2)/(frob(T_true)^2);
        end

        function error = NAE(T_true, T_estimated, R)
            error = (1/R)*sum(abs(T_true/sum(abs(T_true),'all') - T_estimated/sum(abs(T_estimated),'all')), 'all');
        end

        function error = Cost(T_true, T_estimated, Om)
            % Get the objective value for the masked entries
            %   Masks T_true and T_estiamted by Om and returns the frobenius norm.
            error = frob(Om.*(T_true - T_estimated));
        end

    end
end