function test_codebook()
% Run: addpath('path/to/fd-pixel-codebook'); test_codebook
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
results.cfg.center.frequencyHz = 2.4e9;
results.originalLoadPorts = [3 5 7];
results.fullArchitecture = [1 2 0];
results.environmentS21 = [0.1; -0.1];
results.evaluation.stateBits = [0; 1];
S = complex(zeros(2,2,1,2));
S(1,1,1,:) = 0.1; S(2,2,1,:) = 0.1;
S(2,1,1,1) = -0.1; S(2,1,1,2) = 0.1;
results.evaluation.centerS = S;
save(fullfile(folder,'results.mat'),'results');
c.resultsFile = fullfile(folder,'results.mat');
c.maxCodewords = 2; c.outputFolder = fullfile(folder,'out');
c.residualLimitDb = -30;
m = train_fd_pixel_codebook(c);
assert(isequal(m.sourceStateRows,[1;2]));
assert(all(m.assignedCodeword == [1;2]));
assert(m.coverageFraction == 1);
assert(isequal(m.physicalStates,[1 0 0;1 1 0]));
manifest = readtable(fullfile(c.outputFolder,'codebook_cst_states.csv'));
assert(isequal(manifest.Load002_Port5,[0;1]));
lookup = apply_fd_pixel_codebook(fullfile(c.outputFolder,'codebook.mat'),-0.1);
assert(lookup.codeword == 2 && lookup.residualDb == -Inf);
disp('Codebook tests passed.');
end
