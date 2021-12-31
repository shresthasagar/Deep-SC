function T_recovered = full_tps(T, Ovec)
    [I,J,K] = size(T);
    T_recovered = zeros(I,J,K);
    X_coord = [];
    Y_coord = [];
    Z = [];
    O_mat = reshape(Ovec, [I,J]);
    count = 1;
    Z = zeros(1,K);
    for j=1:J
        for i=1:I
            if O_mat(i,j)
                X_coord = [X_coord; i];
                Y_coord = [Y_coord; j];
                Z(count,:) = T(i,j,:);
                count = count + 1;
            end
        end
    end
    x_grid = 1:I;
    y_grid = 1:J;
    [x,y] = meshgrid(y_grid, x_grid);
    lambda = 0.5e1;
    Linv = get_tps_inv(X_coord, Y_coord, lambda);

    kf=@(r) r.^2.*log(abs(r)+eps);

    xvec = x(:);
    yvec = y(:);
    for k = 1:K
        Zvec = Z(:,k);
        Zvec = Zvec(:);
        bmat=[Zvec;zeros(3,1)];
        
        xmat = Linv*bmat;
        w=xmat(1:length(X_coord));
    
        a=xmat(end-2:end);

        GK = kf( sqrt(  (bsxfun(@minus,xvec,X_coord')).^2+(bsxfun(@minus,yvec,Y_coord')).^2  ) );
        N = [ones(size(xvec)),xvec,yvec];
        zvec = GK*w + N*a;
        z = reshape(zvec,size(x));
     
        T_recovered(:,:,k) = reshape(z, [I J])';
    end
end
