function [Xbtd, Sbtd, Cbtd] = btd(T, Strue, Ctrue, Ov)
    use_dB = false;
    %% BTD
    [I J K] = size(T);
    [~, R] = size(Ctrue);

    Sc = zeros([I J R]);
    for rr=1:R
        Sc(:,:,rr) = Strue{rr};
    end

    [I,J,K] = size(T);
    IJ = I*J;
    % num_samples = round(f*IJ);
    % Omega = randperm(IJ, num_samples)';

    % Mode-3 matrix unfolding, arrange fibers as columns of the matrix from tensor
    Tm = tens2mat(T,3);

    % % sampling matrix
    % Ov = false(1,IJ);
    % Ov(Omega) = true;
    O_mat = reshape(Ov, [I J]);
    Omega = find(O_mat);


    gridLen = I-1;
    gridResolution = 1;%ja 
    x_grid = [0:gridResolution:gridLen];
    y_grid = [0:gridResolution:gridLen];
    [Xmesh_grid, Ymesh_grid] = meshgrid(x_grid, y_grid);
    Xgrid = Xmesh_grid + 1i*Ymesh_grid;

    L0 = 4;
    L = L0*ones(1,R);
    sumL = sum(L);

    wind = zeros(I,J);
    wind(Omega)=1;
    
    % for ii = 1:I
    %     wind(ii,ii) = 1;
    % end

    W = zeros(I,J,K); % sampling Omega tensor
    for kk = 1:K
        W(:,:,kk) = wind;
    end

    X = T;

    W1 = tens2mat(W,[],1);
    W2 = tens2mat(W,[],2);
    W3 = tens2mat(W,[],3);

    Y = W.*X;
    Y1 = tens2mat(Y,[],1);
    Y2 = tens2mat(Y,[],2);
    Y3 = tens2mat(Y,[],3);


    Uhat = ll1(Y,L);
    A =[];
    C= [];
    B = [];
    for rr=1:R
        C = [C,Uhat{rr}{3}];
        B = [B,Uhat{rr}{2}];
        A = [A,Uhat{rr}{1}];
    end
    % C = ColumnNormalization(C);

    % Cpderr_init = cpderr(Ctrue,C)
    Cc = Mat2Cell(C,ones(1,R));
    Bc = Mat2Cell(B,L);
    Ac = Mat2Cell(A,L);
    % % for rr = 1:R
    % %
    % %     Bc{rr} = qr(Bc{rr},0);
    % % end
    % % B = Cell2Mat(Bc);

    MaxIter = 50;
    % %     Somega = [S1(Omega'),S2(Omega')];
    Somega = [];
    for rr=1:R
        Scr = Sc(:,:,rr);
        Somega = [Somega,Scr(Omega)];
        
    end
    Xomega = Somega*Ctrue';
    %%
    % lambda = 8e-3;
    % mu = 8e-3;

    lambda = 1e-2;
    mu = 1e-2;

    tic


    iitt = 1

    old_cost = 1000;
    cost = 100;
    % while ~(iitt > MinIter && NAEC-naec_old>0) && iitt < MaxIter
    while iitt< MaxIter && old_cost - cost > 0.001
        % naec_old = NAEC
        %     lam bda = lambda*0.9;
        %     mu = mu*0.9;
        % if iitt>6
        %     lambda = 1e-10;
        %     mu = 1e-10;
        % end
        old_cost-cost
        old_cost = cost;
        
        CpB = PartKron(Cc,Bc);
        % leftA = Y1(:,ii)'*CpB;
        leftA = Y1'*CpB;
        parfor ii=1:I
            rightA = CpB'*diag(W1(:,ii))*CpB + lambda*eye(sumL);
            A(ii,:) = leftA(ii,:)/rightA;
        end
        
        Ac = Mat2Cell(A,L);
        CpA = PartKron(Cc,Ac);
        % leftB = Y2(:,jj)'*CpA ;
        leftB = Y2'*CpA ;
        parfor jj=1:J
            rightB = CpA'*diag(W2(:,jj))*CpA+ lambda*eye(sumL);
            B(jj,:) = leftB(jj,:)/rightB;
        end
        Bc = Mat2Cell(B,L);
        % % % % %     X3= M*C';
        M = zeros(I*J,R);
        
        parfor rr = 1:R
            M(:,rr) = reshape(Ac{rr}*Bc{rr}',[],1);
        end
        % %     parfor kk = 1:K
        % %         C(kk,:) = Y3(:,kk)'*diag(W3(:,kk))*M/(M'*diag(W3(:,kk))*M);
        % %     end
        %             Cold_n = ColumnNormalization(C);
        C = Y3'*M/(M'*diag(W3(:,1))*M+ mu*eye(R));
        C = ColumnNormalization(C);
        
        Cc = Mat2Cell(C,ones(1,R));
        
        X_recons = zeros(I,J,K);
        for rr=1:R
            X_recons = X_recons + outprod(Ac{rr}*Bc{rr}', C(:,rr));
        end
        cost = metric.Cost(T, X_recons, O_mat)
        iitt = iitt + 1
    end
    
    
    %% remove permutation
    [cpderrc,per,~]=cpderr(Ctrue,C);
    % % % CPDERR = cpderrc^2;
    % % % CPDERR_dB = 10*log10(CPDERR)
    C_noperm = C*per;
    
    % error
    Ctrue_n = ColumnNormalization(Ctrue);
    C_p = ColumnPositive(C_noperm);
    C_p(C_p<0)=0;
    C_n = ColumnNormalization(C_p);
    
    
    naec = 0;
    for rr = 1:R
        norm_1c = Ctrue_n(:,rr)/sum(abs(Ctrue_n(:,rr))) - C_n(:,rr)/sum(abs(C_n(:,rr)));
        naec = naec + sum(abs(norm_1c));
    end
    NAEC = naec/R

    Somega_est= Xomega/C_n';
    % % % % % % % %         Somega_est1= Xomega/(C_n+1e-10*eye(size(C_n)))';
    % % % % % % % %         Somega_est2= Xomega/(C_n+1e-19*eye(size(C_n)))';
    % % % % % % % %         frob(Somega_est2-Somega_est)
    for rr = 1:R
        minSc{rr} = min(Sc(:,:,rr),[],'all');
    end

    % % % Soe =ColumnPositive( Somega_est );
    % % % Soe(Soe<0)=0;
    % % % Soe = ColumnNormalization(Soe);
    Soe = Somega_est;
    Soe(Soe<minSc{1}) = minSc{1};
    for rr = 1:R
        so{rr} = Soe(:,rr);
    end

    x_IND = real(Xgrid(Omega));
    y_IND = imag(Xgrid(Omega));
    xgrid = real(Xgrid(:));
    ygrid = imag(Xgrid(:));
    lambda_tps = 1e-4;
    Xbtd = zeros(I,J,K);

    % % Shatvec = [];
    for rr=1:R
        
        stps{rr} = interp(x_IND,y_IND,so{rr},xgrid,ygrid,lambda_tps,true);
        
        Shat{rr} = reshape(stps{rr},I,J);
        Xbtd = Xbtd + outprod(Shat{rr}, C_n(:,rr));
        % %      Shatvec = [Shatvec,Shat{rr}(:)];
        Shat{rr} = Shat{rr}/norm(Shat{rr},'fro');
        Shat{rr}(Shat{rr}<minSc{rr}) = minSc{rr};
    end

    Sbtd = zeros(I,J,R);
    Cbtd = C_n;
    for rr=1:R
        Sbtd(:,:,rr) = Shat{rr};
    end

end