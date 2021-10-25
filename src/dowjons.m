function [X_est, time, S_est, C_est, Z_est] = dowjons(X_omega, Ovec, R, C_true, varargin)
    % Implementation of Nasdac algorithm for completing radio maps
    %   @params: 
    %       X_omega: Tensor of measurements of size [I J K]
    %       Ovec: measurement location indices
    %       R: Number of emitters
    %       C_true: Required for correcting permutation

    % initialize default params
    [I,J,K] = size(X_omega);
    max_iter = 10;
    relative_error_stop = 0.003;
    Om = reshape(Ovec,[I,J]);
    Xm = tens2mat(X_omega, 3);
    S_omega = zeros(R, I*J);
    lambda_c = 0; % optional regularization parameter
    debug_mode = false;
    model_path = 'default';

    % Read the optional parameters
    if (rem(length(varargin),2)==1)
        error('Optional parameters should always go by pairs');
    else
        for i=1:2:(length(varargin)-1)
            switch upper(varargin{i})
                case 'max_iter'
                    max_iter = varargin{i+1};
                case 'relative_error_stop'
                    relative_error_stop = varargin{i+1};
                case 'debug_mode'
                    debug_mode = varargin{i+1};
                case 'MODEL_PATH'
                    model_path = varargin{i+1};
                otherwise
                    % Hmmm, something wrong with the parameter string
                    error(['Unrecognized option: ''' varargin{i} '''']);
            end;
        end;
    end;

    %% Start joint Optimization of C and S
    W = zeros(I*J, I*J);
    for i=1:I*J
        if Ovec(i)
            W(i,i) = 1;
        end
    end
    iter=0;
    previous_loss = 9999;
    
    % initialize using nasdac
    [X_est, time_init, S_est, C_est, Z_est] = nasdac(X_omega, Ovec, R, C_true, 'estimate_z', true, 'model_path', model_path);
    sample_loss = [metric.Cost(X_omega, X_est, Om)];
    time = time_init;
    Sm = tens2mat(S_est, 3);
    Z_est_py = py.numpy.array(Z_est); % current estimate of Z
    Y = Xm*W; % from measurement
    % maintain python numpy arrays to for NN optimization step
    W_py = py.numpy.array(Om);
    X_py = py.numpy.array(X_omega); % measurements

    tic
    while (iter < max_iter) && (previous_loss-sample_loss(end) > relative_error_stop)
        previous_loss = sample_loss(end);
        iter = iter+1;
        %% Step 1
        % Cr optimization subprobplem.
        % Non Linear Least Squares
        
        Q = Sm*W; % estimated S
        
        % with regularization
        C_est = [];
        lambdaI = lambda_c*eye(R);
        A = [Q'; lambdaI;];
        for k=1:K
            b = [(Y(k,:))';zeros(R,1)];
            % c = lsqnonneg(Q', (Y(k,:))');
            c = lsqnonneg(A, b);
            C_est = [C_est; c'];
        end
        % C = ColumnNormalization(C);


        if debug_mode
            X_est = helper.get_tensor(S_est, C_est);
            expected_loss = [expected_loss, metric.SRE(T_true, X_est)];
        end
        
        %% Step 2
        % Stheta optimization subproblem

        % conversion from matlab array to python numpy array.. does not need to be timed as this can be avoided by implementing everything in python        
        time = time + toc;
        C_est_py = py.numpy.array(C_est); % current estimate of C
            
        
        % Call the nn gradient descent optimizer: returns optimized S_omega
        tuple = py.slf_interface.optimize_s(W_py, X_py, Z_est_py, C_est_py, R, 0, model_path);
        cell_tuple = cell(tuple);
        Z_est_py = cell_tuple{1};
        S_est = double(cell_tuple{2});
        time = time + double(cell_tuple{3});
        tic;
        S_omega = S_est.*Om;
        Sm = tens2mat(S_est,3);
        X_est = helper.get_tensor(S_est, C_est);
        sample_loss = [sample_loss, metric.Cost(X_omega, X_est, Om)];
    end
    disp(['dowjons iter', num2str(iter)]);
end