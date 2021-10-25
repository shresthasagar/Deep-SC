clear;

addpath ../src

reload_module

K = 64;
R = 3;
shadow_sigma = 8;
d_corr = 50;
structured_psd = false;
use_snr = false;
rho = 0.05; % sampling fraction

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

% [X_deep, time_deep] = deepcomp(X_true, Ov);
% [X_btd, S_btd, C_btd] = btd(X_true, Sc, C_true, Ov);
% X_tps = full_tps(X_true, Ov);
