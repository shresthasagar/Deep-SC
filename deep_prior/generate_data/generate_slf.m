% Code to generate SLF for training generative model 

% Change the following paths to the location where you want to save the SLF images and the parameters details
IMG_DIR = '../dataset/train_slf_mat/';
DETAILS_FILE = '../dataset/details.csv';

if exist(IMG_DIR, 'dir') ~= 7
    mkdir(IMG_DIR);
end

if exist(DETAILS_FILE, 'file') 
    delete(DETAILS_FILE);
end

% Change the following parameters as required
seed = sum(100*clock);
s = RandStream('mt19937ar','Seed',seed,'NormalTransform','Polar');
loss_f = @(x,d,alpha) min(1,(x/d).^(-alpha));
d0 = 2;
gridLen = 50;       % Size of the SLF length
gridResolution = 1;    
num_examples = 50000;  % Number of SLF images to generate
x_grid = [0:gridResolution:gridLen];
y_grid = [0:gridResolution:gridLen];
[Xmesh_grid, Ymesh_grid] = meshgrid(x_grid, y_grid);
Xgrid = Xmesh_grid + 1i*Ymesh_grid;
[I,J] = size(Xgrid);

fid = fopen(DETAILS_FILE, 'a');
heading = 'filename,Xr,Yr,alpha,Xc,ShadowSigma\n';
fprintf(fid, heading);

for count=1:num_examples
    % change the range of parameters as required
    shadow_sigma = 3 + 6*rand(s);
    Xc = 50 + 10*rand(s);
    
    location = gridLen*rand(s) + 1i*gridLen*rand(s);
    loss_mat = abs(Xgrid - location);
    alpha = 2+0.5*rand(s);
    p = exp(-1/Xc);
    shadow = Shadowing_data(Xgrid,shadow_sigma,p);
    shadow_linear = 10.^(shadow/10);
    Sc = loss_f(loss_mat,d0,alpha).*shadow_linear;
    Sc = Sc/norm(Sc,'fro');

    filename = sprintf('%07d.mat', count);
    newline = strcat(filename, ',', string(real(location)), ',', string(imag(location)), ',', string(alpha), ',', string(Xc), ',', string(shadow_sigma),'\n');
    fprintf(fid, newline);
    save(strcat(IMG_DIR, filename), 'Sc');
    count
end
fclose(fid);
