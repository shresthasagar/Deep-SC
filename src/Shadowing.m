function shadowing_Correlation = Shadowing( Cloc,var,p)
    seed = sum(100*clock);
    s = RandStream('mt19937ar','Seed',seed,'NormalTransform','Polar');
    
    % UNTITLED generate shadowing fading with
    % correlation E(z(x)z(x')) = var^2*exp(-|x-x'|/Xc)
    
    if nargin <3
        p = exp(-1/50);
    end

    %%%Cloc=X1; var = 4;Xc=50;
    if var==0
        shadowing_Correlation=zeros(size(Cloc));
    else
        [m,n] = size(Cloc);
        vec_Cloc = Cloc(:);
        shadowing_iid = var*randn(s,m,n);
        Vec_shadowing_iid = shadowing_iid(:);
        R = @(d) p.^d;
        Distance_Corr = abs(bsxfun(@minus,vec_Cloc , transpose(vec_Cloc ))  );
        S = chol(R(Distance_Corr),'lower');
        Vec_shadowing_Correlation = S*Vec_shadowing_iid;
        shadowing_Correlation = reshape(Vec_shadowing_Correlation, [m n]);
    end
    
end
    