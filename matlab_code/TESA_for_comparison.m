%% 1. Setup and Load Data
% Clear workspace and initialize
clear; close all; eeglab;

filename = '2024-12-17T144344'
output_dir = '/'

% Define paths and parameters
dataPath = '/';
sessionPhaseNumber = 1, 2, 3, 4;
chans = '';

% Get EEGLAB directory and setup channel locations file path
eeglabPath = fileparts(which('eeglab'));
% Use standard 10-20 montage file instead
chanlocFile = fullfile(eeglabPath, 'plugins', 'dipfit', ...
    'standard_BEM', 'elec', 'standard_1020.elc');

% Verify channel location file exists
%if ~exist(chanlocFile, 'file')
   % error('Channel location file not found: %s', chanlocFile);
%end

% Load EEG data
EEG = pop_readneurone(dataPath, sessionPhaseNumber, chans);


% Store original data
ALLEEG = EEG;


EEG = eeg_checkset(EEG);


% Visualize channel locations to verify
%figure;
%topoplot([], EEG.chanlocs, 'style', 'blank', ...
  %  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
%title('Standard 10-20 Channel Locations');

% Remove unused EMG channel 

EEG = pop_select( EEG, 'rmchannel',{'EMG1'});

EEG = eeg_checkset(EEG);


disp("Step 1 completed")
% 2. Remove Bad Channels (Step 1 in TESA pipeline)
% GUI: Tools > Reject data using clean_rawdata

EEG = pop_rejchan(EEG, 'elec', [1:32], ...
    'threshold', 2, ...
    'norm', 'on', ...
    'measure', 'kurt');

EEG = eeg_checkset(EEG);


% Visualize rejected channels
%figure; 
%topoplot([], EEG.chanlocs, 'style', 'blank', ...
   % 'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
%title('Channel Locations After Rejection');

disp("Step 2 completed")
 3. Epoching and Demeaning (Steps 2-3 in TESA pipeline)
% GUI: Tools > Extract Epochs
EEG = pop_epoch(EEG, {'A - Stimulation'}, [-0.4 0.4], ... % -400 to 400 
    'newname', 'TMS_epochs', ...
    'epochinfo', 'yes');

% Remove baseline (GUI: Tools > Remove baseline)
EEG = pop_rmbase(EEG, [-400 -50], []); % -400 to -50 ms baseline as specified

EEG = eeg_checkset(EEG);


% Visualize epochs 
eegplot(EEG.data,  'title', 'Epoched Data', 'events', EEG.event);

disp("Step 3 completed")
% 4. TMS Artifact Removal (Steps 4, 7, 9 in TESA pipeline)
% GUI: Tools > TESA > TMS-EEG Signal Processing
% Remove and interpolate TMS pulse artifact
EEG = tesa_removedata(EEG, [-2 5]); % -2 to 5 ms 
EEG = tesa_interpdata(EEG, 'cubic', [1 1]);

EEG = eeg_checkset(EEG);



% Visualize after TMS artifact removal
%figure;
%plot(EEG.times, mean(mean(EEG.data, 3), 1));
%title('Data After TMS Artifact Removal');
%xlabel('Time (ms)'); ylabel('Amplitude (µV)');

% Visualize epochs
eegplot(EEG.data,  'title', 'Epoched Data', 'events', EEG.event);

disp("Step 4 completed")


% 6. Bad Trial Removal (Step 6 in TESA pipeline)
% GUI: Tools > Reject data epochs
EEG = pop_jointprob(EEG, 1, [1:size(EEG.data,1)], 2, 2, 0, 0, 0, [], 0);
EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1);

EEG = eeg_checkset(EEG);


% Visualize rejected trials
%eegplot(EEG, 1, 1, 1);
%title('Data After Trial Rejection');

disp("Step 6 completed")

% 7. First ICA for TMS Muscle Artifact (Step 8 in TESA pipeline)
% First ICA with detailed parameters

EEG = tesa_fastica(EEG, ...
    'approach', 'symm', ...    % Symmetric approach
    'g', 'tanh', ...          % Nonlinearity
    'stabilization', 'off');

% Component selection with detailed parameters
EEG = tesa_compselect(EEG, ...
    'compCheck', 'on', ...
    'remove', 'on', ...
    'saveWeights', 'off', ...
    'figSize', 'medium', ...
    'plotTimeX', [-100 250], ...
    'freqScale', 'log', ...
    'tmsMuscle', 'on', ...
    'tmsMuscleThresh', 2, ...
    'tmsMuscleWin', [11 30], ...
    'tmsMuscleFeedback', 'off', ...
    'blink', 'off', ...
    'blinkThresh', 2.5, ...
    'blinkElecs', {'Fp1','Fp2'}, ...
    'blinkFeedback', 'off', ...
    'move', 'off', ...
    'moveThresh', 2, ...
    'moveElecs', {'F7','F8'}, ...
    'moveFeedback', 'off', ...
    'muscle', 'off', ...
    'muscleThresh', -0.31, ...
    'muscleFreqIn', [7 70], ...
    'muscleFreqEx', [48 52], ...
    'muscleFeedback', 'off', ...
    'elecNoise', 'off', ...
    'elecNoiseThresh', 2, ...
    'elecNoiseFeedback', 'off');

EEG = eeg_checkset(EEG);




% Visualize rejected trials
%eegplot(EEG, 1, 1, 1);
%title('Data After first ICA');

disp("Step 7 completed")

% 8. Filtering (Step 10 in TESA pipeline)
% GUI: Tools > Filter the data
% Band-pass filter
EEG = tesa_filtbutter(EEG, 0.1, 45, 4, 'bandpass'); % 0.1-45 Hz as specified
% Notch filter
EEG = tesa_filtbutter(EEG, 48, 52, 4, 'bandstop'); % 48-52 Hz as specified

disp("Step 8 completed")

% 9. Second ICA for Remaining Artifacts (Step 12 in TESA pipeline)
% Second ICA using runica
EEG = tesa_fastica(EEG, ...
    'approach', 'symm', ...    % Symmetric approach
    'g', 'tanh', ...          % Nonlinearity
    'stabilization', 'off');

% Component selection focusing on blink and movement
try
    EEG = tesa_compselect(EEG, ...
        'compCheck', 'on', ...
        'remove', 'on', ...
        'saveWeights', 'off', ...
        'figSize', 'medium', ...
        'plotTimeX', [-100 250], ...
        'plotFreqX', [1 100], ...
        'freqScale', 'log', ...
        'tmsMuscle', 'off', ...
        'blink', 'on', ...
        'blinkThresh', 2.5, ...
        'blinkElecs', {'Fp1','Fp2'}, ...
        'blinkFeedback', 'off', ...
        'move', 'on', ...
        'moveThresh', 2, ...
        'moveElecs', {'F7','F8'}, ...
        'moveFeedback', 'off', ...
        'muscle', 'on', ...
        'muscleThresh', -0.31, ...
        'muscleFreqIn', [7 70], ...
        'muscleFreqEx', [48 52], ...
        'muscleFeedback', 'off', ...
        'elecNoise', 'on', ...
        'elecNoiseThresh', 2, ...
        'elecNoiseFeedback', 'off');
catch
    % If channels missing, interpolate first
    %disp('Interpolating channels before component selection...');
    %EEG = interp(EEG, ALLEEG(1).chanlocs, 'spherical');
    EEG = tesa_compselect(EEG, ...
        'compCheck', 'on', ...
        'remove', 'on', ...
        'blink', 'on', ...
        'move', 'on', ...
        'muscle', 'on');
end

disp("Step 9 completed")


%% 10. Final Steps (Steps 14-16 in TESA pipeline)
% Interpolate bad channels
EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');

% Rereference to average
EEG = pop_reref(EEG, []);

% Final baseline correction
EEG = pop_rmbase(EEG, [-400 -50], []);

disp("Step 10 completed")

%%
eegplot(EEG.data,  'title', 'Epoched Data', 'events', EEG.event);

% Save processed EEG data
EEG = pop_saveset(EEG, 'filename','processed__neo_data.set', 'filepath', output_dir)

%% 11. Visualization and Analysis
EEG = tesa_tepextract( EEG, 'GMFA' );
EEG = tesa_peakanalysis( EEG, 'GMFA', 'positive', [30 55 180], [20 40;45 65;160 200], 'method' ,'largest', 'samples', 5, 'tepName', 'R1' );
EEG = tesa_peakanalysis( EEG, 'GMFA', 'negative', [45 100], [35 55;85 115], 'method' ,'largest', 'samples', 5, 'tepName', 'R1' );
output = tesa_peakoutput( EEG, 'tepName', 'R1', 'calcType', 'amplitude', 'winType', 'individual', 'averageWin', [], 'fixedPeak', [], 'tablePlot', 'on' );
tesa_plot( EEG, 'tepType', 'GMFA', 'tepName', 'R1', 'xlim', [-300 300], 'ylim', [], 'CI','off','plotPeak','on' );
tesa_plot( EEG, 'tepType', 'data', 'xlim', [-300 300], 'ylim', [], 'CI','off','plotPeak','off' );



%plot all channels averaged
figure(100)
timeVector = EEG.xmin:1/EEG.srate:EEG.xmax;
nChans = size(EEG.data,1);
nCols = 4;
nRows = ceil(nChans/nCols);
sgtitle('Channel average')
for i = 1:nChans
    chan_avg = mean(squeeze(EEG.data(i,:,:)),2);
    subplot(nRows,nCols,i)
    plot(timeVector,chan_avg)
    title(EEG.chanlocs(i).labels)
    xlabel('time [ms]')
    ylabel('Potential [\mu s]')
end

%%
[num_channels, num_time_points, num_trials] = size(EEG.data);
disp(['EEG Data Dimensions: ', num2str(num_channels), ' channels x ', num2str(num_time_points), ' time points x ', num2str(num_trials), ' trials']);

% Ensure there are multiple channels
if num_channels < 2
    error('PCI analysis requires data from multiple channels. Only %d channel(s) detected.', num_channels);
end

times = EEG.times; % Assuming EEG.times is [1 x time_points] or [time_points x 1]
disp(['Time Vector Length: ', num2str(length(times)), ' ms']);


%% 12. PCI Analysis
% Calculate PCI on the processed data

% Get the data ready - average across trials
avg_data = mean(EEG.data, 3); % Average across trials
times = EEG.times; % Time vector in ms

% Set up parameters
parameters = [];
parameters.max_var = 99;              % Percentage of variance for PCA
parameters.min_snr = 1.1;             % Minimum signal-to-noise ratio
parameters.k = 1.2;                   % Noise control parameter
parameters.baseline = [-400 -50];      % baseline period [ms]
parameters.response = [0 300];        % response period [ms]
parameters.nsteps = 100;             % Number of steps for threshold search
parameters.l = 1;                    % Number of embedding dimensions
parameters.tau = 2;                  % Embedding delay

% Run PCIst
[pci, dNST, parameters] = PCIst(avg_data, times, parameters);

disp(pci)