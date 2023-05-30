function [X Sc Ctrue peaks Bmat] = generate_map(dB, K, R, shadow_sigma, Xc, structured_c, basis, strictly_separable, spatial_resolution)
    seed = sum(100*clock);
    s = RandStream('mt19937ar','Seed',seed,'NormalTransform','Polar');
    
    indK = [1:K]';
    if nargin <7
        basis = 's'; 
    end
    
    if basis == 'g'
        Sx = @(f0,sigma) exp(-(indK-f0).^2 /(2*sigma^2)); 
    else
        Sx =@(f0,a) sinc((indK-f0)/a).^2.*( abs((indK-f0)/a)<=1);
    end

    Ctrue=[];
    num_peaks_per_psd = 16;

    ind_psd = R*3+3:2:K-2;
    Bmat = [];
    if strictly_separable
        for rr=1:R
            psd_peaks = ind_psd(randperm(length(ind_psd), num_peaks_per_psd));
            % am = 0.5+ 1.5*rand(num_peaks_per_psd,1);
            am = 0.5 + 1.5*rand(num_peaks_per_psd,1);

            % First peak, ensure separability if needed
            % c{rr} = am(1)*Sx(rr,1);
            % c{rr} = am(1)*Sx(1+(rr-1)*3,2+3*rand());
            c{rr} = am(1)*Sx(1+(rr-1)*2.5,2+3*rand());

            Bmat = [Bmat c{rr}];
            % c{rr} = am(1)*Sx(psd_peaks(1),2+2*rand());
            
            % Remaining peaks
            for q=2:num_peaks_per_psd-1
                c{rr} = c{rr} + am(q)*Sx(psd_peaks(q),2+2.5*rand());
            end
            Bmat = [Bmat c{rr}];
            Ctrue = [Ctrue,c{rr}];
        end
    else
        psd_peaks = ind_psd(randperm(length(ind_psd), num_peaks_per_psd-1));            
        for rr=1:R
            am = 0.5 + 1.5*rand(num_peaks_per_psd,1);

            % First peak, ensure separability if needed
            c{rr} = am(1)*Sx(2+(rr-1)*2.2,2+2.3*rand());
            Bmat = [Bmat c{rr}];

            % Remaining peaks
            for q=1:num_peaks_per_psd-1
                c{rr} = c{rr} + am(q)*Sx(psd_peaks(q),2+2*rand());
            end
            Bmat = [Bmat c{rr}];
            Ctrue = [Ctrue,c{rr}];
        end
    end
    
    Ctrue = ColumnNormalization(Ctrue);

    loss_f = @(x,d,alpha) min(1,(x/d).^(-alpha));
    d0 = 2;

    gridLen = 50;
    gridResolution = 1;%ja 
    x_grid = [0:gridResolution:gridLen];
    y_grid = [0:gridResolution:gridLen];
    [Xmesh_grid, Ymesh_grid] = meshgrid(x_grid, y_grid);
    Xgrid = Xmesh_grid + 1i*Ymesh_grid;
    % X1 = transpose(X1);
    [I,J] = size(Xgrid);


    Svec = [];
    % locations = [5+5i, 20+5i, 35+5i, 48+5i, 10+20i, 30+20i, 45+20i, 5+35i, 40+35i,  2+48i, 48+48i, 22+29i, 32+47i, 16+45i];

    % locations = locations(randperm(length(locations), R));
    if nargin < 9
        spatial_resolution = 50;
    end

    spatial_offset = 1 + (50 - spatial_resolution)/2;
    
    for rr=1:R
        % location = 10+30*rand() + 1i*(10+30*rand());
        location = (spatial_offset + spatial_resolution*rand()) + 1i*(spatial_offset + spatial_resolution*rand());
        peaks{rr} = [uint8(real(location)), uint8(imag(location))];
        % location = locations(rr);
        loss_mat = abs(Xgrid - location);
        alpha = 2+0.5*rand;
        p = exp(-1/Xc);
        shadow = Shadowing(Xgrid,shadow_sigma,p);
        shadow_linear = 10.^(shadow/10);
        Sc{rr} = loss_f(loss_mat,d0,alpha).*shadow_linear;
        Sc{rr} = Sc{rr}/norm(Sc{rr},'fro');
        Svec = [Svec,Sc{rr}(:)];
    end

    if dB
        for rr=1:R
            Sc{rr} = real(10*log10(Sc{rr}));  % Remove all the fibers with nan entries along the column
        end
    end

    X = zeros(I,J,K);
    for rr=1:R
        X = X + outprod(Sc{rr},Ctrue(:,rr));
    end


    
end