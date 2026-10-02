% Run this script from any folder. Set seboResultFile to your saved run.
thisFolder = fileparts(mfilename('fullpath'));
addpath(thisFolder);
seboResultFile = fullfile(thisFolder,'input','results.mat'); % EDIT THIS PATH

% Complex S21 environment cloud: about -20 to -15 dB.
[x,y] = ndgrid(-1:0.05:1,-1:0.05:1);
z = complex(x(:),y(:));
z = z(abs(z) >= 10^(-5/20) & abs(z) <= 1);
cloudS21 = 10^(-15/20)*z;

c = struct();
c.resultsFile = seboResultFile;
c.environmentS21 = cloudS21; % Omit to use the cloud in results.mat.
c.maxCodewords = 20;         % Number of stored diode states, <= eligible states.
c.selectionMetric = 'COVERAGE'; % COVERAGE, MINIMAX, or MEAN.
c.residualLimitDb = -20;
c.matchEnabled = true;       % Set false to ignore S11/S22 restrictions.
c.limitS11Db = -10;
c.limitS22Db = -10;
c.outputFolder = fullfile(thisFolder,'codebook_results');
model = train_fd_pixel_codebook(c);

% The lookup also accepts a path to codebook.mat.
lookup = apply_fd_pixel_codebook(model,cloudS21(1:5));
disp(table((1:5).',lookup.codeword(:),lookup.residualDb(:), ...
    'VariableNames',{'Point','Codeword','ResidualDb'}));
