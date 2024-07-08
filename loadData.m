function [eeg,envelopes,attSpeaker,fs] = loadData(dataset,subject,preprocessing)
% LOADDATA Load the EEG and attended speaker of a given subject and
% dataset. The trials are preprocessed with the given parameters in
% preprocessing (normalization, rereferencing and channel selection).
% Potentially, conditions to select can be given (variable input).
%
%   Input parameters:
%       dataset [STRING]: dataset name
%       subject [INTEGER]: subject number to load the data from
%       preprocessing [STRUCT]: preprocessing structure with normalization
%                               field (binary), rereferencing field ('none','Cz','CAR','custom'),
%                               and potential channel selection (vector of channels).

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Load correct data
switch dataset
    case 'das-2016'
        % load data
        load([pwd,'/preprocessed_data/dataS',num2str(subject),'.mat']);
        trialLength = size(eegTrials{1},1);

        % convert trials to tensor
        nbTrials = length(attendedEar);
        eeg = cell2mat(eegTrials);
        eeg = reshape(eeg,[size(eegTrials{1},1),nbTrials,size(eegTrials{1},2)]);
        eeg = permute(eeg,[3 1 2]);
        envelopes = cell2mat(audioTrials);
        envelopes = reshape(envelopes,[size(audioTrials{1},1),nbTrials,size(audioTrials{1},2)]);
        envelopes = permute(envelopes,[3 1 2]);

        cz = 48;
    otherwise
        error('Choose valid dataset')
end

%% preprocessing
eegOrig = eeg;

% channel selection
if ~isempty(preprocessing.eegChanSel)
    eeg = eeg(preprocessing.eegChanSel,:,:);
end

% rereferencing
switch preprocessing.rereference
    case 'none'
    case 'Cz'
        eeg =  eeg - eegOrig(cz,:,:);
    case 'CAR'
        eeg = eeg - mean(eeg(:,:,1));
    case 'custom'
        % for any custom you want
end
clear('eegOrig');

% normalization
if preprocessing.normalization
    for tr = 1:nbTrials
        eeg(:,:,tr) = eeg(:,:,tr) - mean(eeg(:,:,tr),2);
        eeg(:,:,tr) = eeg(:,:,tr)./norm(eeg(:,:,tr),'fro')*size(eeg,2);
    end
end
end