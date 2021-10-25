function z = interp( X,Y,Z,x,y,lambda,dB )
%	Input data: (X,Y)-->Z; vector-form or matrix-form.
%   We use thin-plate splines to interpolate (X,Y)-->Z with smoothing
%   parameter lambda.
%   We want to predict z date given x and y.

    if nargin<7
        dB=0;
    end

    kf=@(r) r.^2.*log(abs(r)+eps);

    % vectorize
    Xvec=X(:);
    Yvec=Y(:);
    Zvec=Z(:);
    if dB
        Zvec=log10(Z(:));
    end
    xvec = x(:);
    yvec = y(:);
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

    bmat=[Zvec;zeros(3,1)];
    % xmat=pinv(Amat)*bmat;
    xmat = Amat\bmat;
    w=xmat(1:len);
    a=xmat(end-2:end);

    GK = kf( sqrt(  (bsxfun(@minus,xvec,Xvec')).^2+(bsxfun(@minus,yvec,Yvec')).^2  ) );
    N = [ones(size(xvec)),xvec,yvec];
    zvec = GK*w + N*a;
    z = reshape(zvec,size(x));
    if dB
        z = 10.^(z);
    end

end

