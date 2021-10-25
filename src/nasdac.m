function [X_est, time, S_est, C_est, Z_est] = nasdac(X_omega, Ovec, R, C_true, varargin)
    % Implementation of Nasdac algorithm for completing radio maps
    %   @params: 
    %       X_omega: Tensor of measurements of size [I J K]
    %       Ovec: measurement location indices
    %       R: Number of emitters
    %       C_true: Required for correcting permutation

    % set default parameters
    normalize_input_columns = true;
    [I,J,K] = size(X_omega);
    S_omega = zeros(R, I*J);
    Om = reshape(Ovec,[I,J]);

    get_Z_est = false;
    model_path = 'default';

    % Read the optional parameters
    if (rem(length(varargin),2)==1)
        error('Optional parameters should always go by pairs');
    else
        for i=1:2:(length(varargin)-1)
            switch upper(varargin{i})
                case 'NORMALIZE_INPUT_COLUMNS'
                    normalize_input_columns = varargin{i+1};
                case 'ESTIMATE_Z'
                    get_Z_est = varargin{i+1};
                case 'MODEL_PATH'
                    model_path = varargin{i+1};
                otherwise
                    % Hmmm, something wrong with the parameter string
                    error(['Unrecognized option: ''' varargin{i} '''']);
            end;
        end;
    end;

    tic;

    Xm_omega = tens2mat(X_omega, 3);
    Xm_omega = Xm_omega(:,Ovec)';

    % normalize input columns
    if normalize_input_columns
        [Xm_omega_norm, Normalizer] = ColumnSumNormalization(Xm_omega);
        Xm_omega = Xm_omega_norm;    
    end

    indices_S = SPA(Xm_omega, R);
    Sm_omega = Xm_omega(:,indices_S);

    % unnormalize input columns
    if normalize_input_columns
        Sm_omega = Sm_omega.*Normalizer(:, indices_S);
        Xm_omega = Xm_omega.*Normalizer;
    end

    % obtain the C matrix 
    pseudo_inverse_S = helper.pseudo_inverse(Sm_omega);
    C = pseudo_inverse_S*Xm_omega;
    C = C';
    t1 = toc;
    
    % remove permutation
    % this is important only if the latent factors need to be recoverd
    [cpderrc,per,~]=cpderr(C_true,C);
    C_noperm = C*per;
    C_p = ColumnPositive(C_noperm);
    C_p(C_p<0)=0;
    [C, d] = ColumnNormalization(C_p);

    Sm_omega = Sm_omega*per;
    Sm_omega = Sm_omega.*d;
    Sm_omega = Sm_omega';
    Xm_omega = Xm_omega';

    tic;
    %% Reconstruct spatial loss field for each emitter from the Sm_omega matrix
    j = 1;
    for i=1:I*J
        if Ovec(i)
            S_omega(:,i) = Sm_omega(:,j);
            j = j+1;
        end
    end
    S_omega = mat2tens(S_omega,[I J R], 3);
    t1 = t1 + toc;

    % prepare data in python for interfacing deep completion network
    W_py = py.numpy.array(Om);
    S_py = py.numpy.array(S_omega);
    
    % output of nasdac-AE 
    tic;
    tuple = py.slf_interface.nasdac_complete(S_py, W_py, R, model_path);
    time = toc + t1;

    % cell_tuple = cell(tuple);
    S_est = double(tuple);
    X_est = helper.get_tensor(S_est, C);
    C_est = C;

    Z_est = 0;
    if get_Z_est
        tuple = py.slf_interface.dowjons_get_initial_z(S_py, W_py, R, model_path);
        cell_tuple = cell(tuple);
        Z_est = cell_tuple{1};
    end

end