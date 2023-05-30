clear;

addpath ../src

reload_module

K = 64;
R = 5;
shadow_sigma = 6;
d_corr = 50;
structured_psd = false;
use_snr = false;
rho = 0.1; % sampling fraction

% Generate map
[X_true, Sc, C_true, peaks, Bmat] = generate_map(false, K, R, shadow_sigma, d_corr, structured_psd, 's', false, 50); 

C_true = ColumnNormalization(C_true);
   
% Add noise
if use_snr
    Ps = frob(X_true)^2;
    Pn = Ps*10^(-snr/10);
    sn = sqrt(Pn/I/J/K);
    if sn>=1e2
        sn =0;
    end
    X_true = X_true + sn*abs(randn(I,J,K));
else
    X_true = X_true;
end
X_true(X_true<0) = 0;

% sample f fraction of psd measurements
[I,J,K] = size(X_true);
IJ = I*J;
num_samples = round(rho*IJ);
Omega = randperm(IJ, num_samples)';

% Vector of sampling indices
Ov = false(1,IJ);
Ov(Omega) = true;

[X_nasdac, time_nasdac, S_nasdac, C_nasdac] = nasdac(X_true, Ov, R, C_true);

[X_dowjons, time_dowjons, S_dowjons, C_dowjons] = dowjons(X_true, Ov, R, C_true);

% Plot figures
k = 5;  % change this if the slab is empty


h = figure(1);

tiledlayout(1, 3, 'Padding', 'none', 'TileSpacing', 'compact'); 
    
nexttile    
contourf(10*log10(X_true(:,:,k)), 100, 'linecolor', 'None');
set(gca,'XTick',[],'YTick',[])
header = sprintf('True Map')
title(header);
colormap jet;


nexttile    
contourf(10*log10(X_nasdac(:,:,k)), 100, 'linecolor', 'None');
set(gca,'XTick',[],'YTick',[])
header = sprintf('Nasdac')
title(header);
colormap jet;

nexttile    
contourf(10*log10(X_dowjons(:,:,k)), 100, 'linecolor', 'None');
set(gca,'XTick',[],'YTick',[])
header = sprintf('Dowjons')
title(header);
colormap jet;

saveas(gcf, 'figure.pdf')
