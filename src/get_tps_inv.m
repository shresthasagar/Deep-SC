function Linv = get_tps_inv( X,Y,lambda)
    %	Input data: (X,Y)-->Z; vector-form or matrix-form.
    %   We use thin-plate splines to interpolate (X,Y)-->Z with smoothing
    %   parameter lambda.
    %   We want to predict z date given x and y.
    
    % % demo  % %
    % % x_grid = 0:2:10;
    % % y_grid = 0:2:10;
    % % [X,Y] = meshgrid(x_grid,y_grid);
    % % Z = 1*sin((X-Y).^2) + 1*randn(size(X));
    % % xi_grid = 0:0.2:10;
    % % yi_grid = 0:0.2:10;
    % % [x,y] = meshgrid(xi_grid,yi_grid);
    % % lambda = 1e-2;
    % % z = TPS( X,Y,Z,x,y,lambda );
    % % mesh(x,y,z);
    % % hold on;plot3(X,Y,Z,'o')
    
    %% function begin 
    % kernel function
    
    kf=@(r) r.^2.*log(abs(r)+eps);
    
    % vectorize
    Xvec=X(:);
    Yvec=Y(:);
    
    len=length(Xvec);
    % K=zeros(len,len);
    
    dist = sqrt((bsxfun(@minus,Xvec,Xvec')).^2+(bsxfun(@minus,Yvec,Yvec')).^2);
    K = kf(dist);
    % for ii=1:len
    %     for jj=1:len
    %         K(ii,jj)=kf(norm([Xvec(ii)-Xvec(jj),Yvec(ii)-Yvec(jj)]));
    %     end
    % end
    P=[ones(size(Xvec)),Xvec,Yvec];
    Amat=[K+lambda*eye(size(K,1)),P;P',zeros(3,3)];
    
    Linv = inv(Amat);
    end