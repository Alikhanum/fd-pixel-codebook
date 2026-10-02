function model = train_fd_pixel_codebook(cfg)
%TRAIN_FD_PIXEL_CODEBOOK Select diode states for complex S21 environments.
% The source is results.mat from scalable_pixel_sebo/run_reduced_port_sebo.
% For each environment point e and diode state q, residual = |e + S21(q)|.
% This trains a center-frequency codebook; no antenna solve is repeated.

assert(nargin == 1 && isstruct(cfg), 'Supply a configuration struct.');
assert(isfield(cfg,'resultsFile') && isfile(cfg.resultsFile), ...
    'cfg.resultsFile must point to a scalable SEBO results.mat.');
source = load(cfg.resultsFile,'results');
assert(isfield(source,'results'), 'Missing results variable.');
r = source.results;
assert(isfield(r,'evaluation') && isfield(r.evaluation,'centerS') && ...
    isfield(r.evaluation,'stateBits') && isfield(r,'fullArchitecture') && ...
    isfield(r,'originalLoadPorts'), 'Missing scalable SEBO result fields.');
ev = r.evaluation;
Q = size(ev.stateBits,1);
% stateBits can enumerate groups; physicalStateBits expands group columns.
if isfield(ev,'physicalStateBits')
    bits = double(ev.physicalStateBits);
else
    bits = double(ev.stateBits);
end
assert(size(bits,1) == Q, 'State rows do not align with centerS.');
assert(ndims(ev.centerS) <= 4 && size(ev.centerS,1) == 2 && ...
    size(ev.centerS,2) == 2 && size(ev.centerS,3) == 1 && ...
    size(ev.centerS,4) == Q, 'Expected 2-by-2-by-1-by-Q centerS.');
a = double(r.fullArchitecture(:).');
p = double(r.originalLoadPorts(:).');
assert(numel(a) == numel(p) && size(bits,2) == nnz(a == 2), ...
    'Diode columns must match fullArchitecture==2 in original load-port order.');
assert(all(bits(:) == 0 | bits(:) == 1), 'Invalid diode state bits.');
s21 = reshape(ev.centerS(2,1,1,:),Q,1);
s11 = reshape(ev.centerS(1,1,1,:),Q,1);
s22 = reshape(ev.centerS(2,2,1,:),Q,1);
assert(all(isfinite(real(s21)) & isfinite(imag(s21))) && ...
    all(isfinite(abs(s11)) & isfinite(abs(s22))), 'Nonfinite S-parameters.');

if ~isfield(cfg,'environmentS21') || isempty(cfg.environmentS21)
    assert(isfield(r,'environmentS21'), 'Supply cfg.environmentS21.');
    cloud = r.environmentS21(:);
else
    cloud = cfg.environmentS21(:);
end
assert(~isempty(cloud) && isnumeric(cloud) && ...
    all(isfinite(real(cloud)) & isfinite(imag(cloud))), ...
    'Environment S21 must be a nonempty finite complex vector.');
cfg = defaults(cfg,'maxCodewords',8);
cfg = defaults(cfg,'selectionMetric','COVERAGE');
cfg = defaults(cfg,'residualLimitDb',-20);
cfg = defaults(cfg,'matchEnabled',true);
cfg = defaults(cfg,'limitS11Db',-10);
cfg = defaults(cfg,'limitS22Db',-10);
cfg = defaults(cfg,'batchSize',256);
cfg = defaults(cfg,'outputFolder',fullfile(fileparts(cfg.resultsFile),'codebook'));
validateattributes(cfg.maxCodewords,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.batchSize,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.residualLimitDb,{'numeric'},{'scalar','real','finite'});
assert(islogical(cfg.matchEnabled) && isscalar(cfg.matchEnabled), ...
    'cfg.matchEnabled must be logical.');
metric = upper(char(cfg.selectionMetric));
assert(ismember(metric,{'COVERAGE','MEAN','MINIMAX'}), ...
    'selectionMetric must be COVERAGE, MEAN, or MINIMAX.');
eligible = true(Q,1);
if cfg.matchEnabled
    eligible = 20*log10(abs(s11)) <= cfg.limitS11Db & ...
        20*log10(abs(s22)) <= cfg.limitS22Db;
end
assert(any(eligible), 'No diode states meet the selected S11/S22 limits.');
assert(cfg.maxCodewords <= nnz(eligible), ...
    'maxCodewords exceeds the number of eligible states.');

% Batch the state dimension: memory scales as environment points * batchSize.
limit = 10^(cfg.residualLimitDb/20);
best = inf(numel(cloud),1);
chosen = zeros(cfg.maxCodewords,1);
coverageCurve = zeros(cfg.maxCodewords,1);
worstCurveDb = zeros(cfg.maxCodewords,1);
meanCurveDb = zeros(cfg.maxCodewords,1);
for k = 1:cfg.maxCodewords
    candidates = find(eligible);
    candidates(ismember(candidates,chosen(1:k-1))) = [];
    score = nan(numel(candidates),3);
    for first = 1:cfg.batchSize:numel(candidates)
        rows = first:min(first+cfg.batchSize-1,numel(candidates));
        trial = abs(bsxfun(@plus,cloud,s21(candidates(rows)).'));
        trial = min(trial,best);
        score(rows,1) = sum(trial <= limit,1).';
        score(rows,2) = mean(trial,1).';
        score(rows,3) = max(trial,[],1).';
    end
    switch metric
        case 'COVERAGE', keys = [-score(:,1),score(:,2),score(:,3)];
        case 'MEAN', keys = [score(:,2),-score(:,1),score(:,3)];
        case 'MINIMAX', keys = [score(:,3),-score(:,1),score(:,2)];
    end
    [~,rank] = sortrows([keys,candidates],[1 2 3 4]);
    chosen(k) = candidates(rank(1));
    best = min(best,abs(cloud+s21(chosen(k))));
    coverageCurve(k) = mean(best <= limit);
    worstCurveDb(k) = 20*log10(max(best));
    meanCurveDb(k) = 20*log10(mean(best));
end

% Stable tie break: first selected codeword wins.
[assigned,residual,word] = assign_points(cloud,s21(chosen));
stateBits = bits(chosen,:);
physicalStates = repmat(a,numel(chosen),1);
physicalStates(:,a == 2) = stateBits;
model = struct('sourceResultsFile',cfg.resultsFile, ...
    'frequencyHz',r.cfg.center.frequencyHz,'configuration',cfg, ...
    'environmentS21',cloud,'sourceStateRows',chosen, ...
    'stateBits',stateBits,'touchstoneLoadPorts',p, ...
    'diodeTouchstonePorts',p(a == 2),'fullArchitecture',a, ...
    'physicalStates',physicalStates,'S21',s21(chosen), ...
    'S11',s11(chosen),'S22',s22(chosen), ...
    'assignedCodeword',word,'assignedS21',assigned, ...
    'residualComplex',cloud+assigned,'residualLinear',residual, ...
    'residualDb',20*log10(residual), ...
    'coverageFraction',mean(residual <= limit), ...
    'worstResidualDb',20*log10(max(residual)), ...
    'meanResidualDb',20*log10(mean(residual)), ...
    'coverageCurve',coverageCurve,'worstCurveDb',worstCurveDb, ...
    'meanCurveDb',meanCurveDb);

folder = cfg.outputFolder;
if ~isfolder(folder), mkdir(folder); end
save(fullfile(folder,'codebook.mat'),'model','-v7.3');
write_exports(model,folder);
make_plots(model,folder,limit);
fprintf('Codebook: %d states, %d points; coverage %.1f%% at %.1f dB, worst %.2f dB.\n', ...
    numel(chosen),numel(cloud),100*model.coverageFraction, ...
    cfg.residualLimitDb,model.worstResidualDb);
fprintf('Saved to %s\n',folder);
end

function cfg = defaults(cfg,name,value)
if ~isfield(cfg,name) || isempty(cfg.(name)), cfg.(name) = value; end
end

function [selected,residual,word] = assign_points(cloud,s21)
residual = inf(size(cloud)); word = zeros(size(cloud));
for k = 1:numel(s21)
    trial = abs(cloud+s21(k));
    improve = trial < residual;
    residual(improve) = trial(improve);
    word(improve) = k;
end
selected = s21(word);
end

function write_exports(m,folder)
K = numel(m.sourceStateRows);
word = (1:K).'; sourceStateRow = m.sourceStateRows;
T = table(word,sourceStateRow,real(m.S11),imag(m.S11), ...
    real(m.S21),imag(m.S21),real(m.S22),imag(m.S22), ...
    20*log10(abs(m.S11)),20*log10(abs(m.S21)),20*log10(abs(m.S22)), ...
    'VariableNames',{'Codeword','SourceStateRow','S11Real','S11Imag', ...
    'S21Real','S21Imag','S22Real','S22Imag','S11dB','S21dB','S22dB'});
for j = 1:numel(m.touchstoneLoadPorts)
    name = sprintf('Load%03d_Port%d',j,m.touchstoneLoadPorts(j));
    T.(name) = m.physicalStates(:,j);
end
writetable(T,fullfile(folder,'codebook_cst_states.csv'));
point = (1:numel(m.environmentS21)).';
A = table(point,real(m.environmentS21),imag(m.environmentS21), ...
    m.assignedCodeword,m.sourceStateRows(m.assignedCodeword), ...
    real(m.residualComplex),imag(m.residualComplex),m.residualDb, ...
    m.residualLinear <= 10^(m.configuration.residualLimitDb/20), ...
    'VariableNames',{'EnvironmentPoint','EnvironmentS21Real', ...
    'EnvironmentS21Imag','Codeword','SourceStateRow','ResidualReal', ...
    'ResidualImag','ResidualDb','MeetsLimit'});
writetable(A,fullfile(folder,'environment_assignments.csv'));
end

function make_plots(m,folder,limit)
fig = figure('Visible','on','Name','Environment cloud and residual');
subplot(1,2,1);
plot(real(m.environmentS21),imag(m.environmentS21),'.','Color',[.75 .75 .75]); hold on;
plot(real(-m.S21),imag(-m.S21),'ko','MarkerFaceColor','r');
axis equal; grid on; xlabel('Real S21'); ylabel('Imaginary S21');
title('Environment and codeword cancellation centers');
legend('Environment','Selected states: -S21','Location','best');
subplot(1,2,2);
pass = m.residualLinear <= limit;
plot(real(m.environmentS21(pass)),imag(m.environmentS21(pass)),'.'); hold on;
plot(real(m.environmentS21(~pass)),imag(m.environmentS21(~pass)),'.');
axis equal; grid on; xlabel('Real environment S21'); ylabel('Imaginary environment S21');
title(sprintf('Residual <= %.1f dB: %.1f%%',m.configuration.residualLimitDb,100*m.coverageFraction));
legend('Meets limit','Exceeds limit','Location','best');
save_pair(fig,folder,'environment_cloud_and_residual');
fig = figure('Visible','on','Name','Codebook training');
plot(1:numel(m.coverageCurve),100*m.coverageCurve,'o-'); grid on;
xlabel('Number of codewords'); ylabel('Environment points covered (%)');
title(sprintf('Coverage at residual <= %.1f dB',m.configuration.residualLimitDb));
save_pair(fig,folder,'codebook_coverage_curve');
end

function save_pair(fig,folder,name)
savefig(fig,fullfile(folder,[name '.fig']));
try
    exportgraphics(fig,fullfile(folder,[name '.png']),'Resolution',180);
catch
    saveas(fig,fullfile(folder,[name '.png']));
end
close(fig);
end
