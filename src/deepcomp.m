function [X_est, time ] = deepcomp(X_omega, Ovec, varargin)
    % Complete the map using trained deepcomp model
    % Read the optional parameters
    model_path = 'default';

    if (rem(length(varargin),2)==1)
        error('Optional parameters should always go by pairs');
    else
        for i=1:2:(length(varargin)-1)
            switch upper(varargin{i})
                case 'MODEL_PATH'
                    model_path = varargin{i+1};
                otherwise
                    % Hmmm, something wrong with the parameter string
                    error(['Unrecognized option: ''' varargin{i} '''']);
            end;
        end;
    end;

    [I J K] = size(X_omega);
    Om = reshape(Ovec,[I,J]);

    W_py = py.numpy.array(Om);
    X_py = py.numpy.array(X_omega);
    
    tic;
    tuple = py.map_interface.map_complete(X_py, W_py, model_path);
    time = toc;
    X_est = double(tuple);
end