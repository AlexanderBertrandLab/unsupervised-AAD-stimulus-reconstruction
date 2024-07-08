%% SUBJECT-SPECIFIC UNSUPERVISED SELF-ADAPTIVE AUDITORY ATTENTION DECODING

% main-file to reproduce experiments starting from random initial decoder on Das2016 dataset.

% scheme:
%   1. update covariance matrix and decoder
%   2. predict attended labels
%   3. update cross-correlation and decoder

% Requires the Tensorlab toolbox for data handling
% (https://www.tensorlab.com/).

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

clear; close all;

%% Setup: parameters
params.dataset = 'das-2016'; % 'das-2016'
params.subjects = 1:16; % subjects to test
params.windowLengths.updating = [60]; % window length to perform updating on
params.windowLengths.test = [60]; % decision window length
params.save = false; % save or not
params.saveName = 'temp'; % name to save results with

% initial decoder - should not matter for beta = 0
params.initialRxx = 'random-full'; % initial Rxx matrix
params.initialRxy = 'random-full'; % initial rxy vector: 'subject-independent', 'random-full', 'random-y', 'random-envelope'

% preprocessing
params.preprocessing.eegChanSel = [];
params.preprocessing.normalization = false;
params.preprocessing.rereference = 'none'; % 'none' / 'Cz' / 'CAR' / 'custom'

% covariance matrix design
params.cov.L = [0,0.25];
params.cov.method = 'lwcov'; % covariance matrix estimation method on new data: 'cov' (no regularization) / 'classic' (ridge regression with predefined hyperparameter) / 'lwcov'
params.cov.lambda = 0; % only applicable if 'cov.method = classic'

% updating parameters
params.updating.iMax = 10; % number of iterations for each updating (1 = one new prediction) to increase convergence speed
params.updating.alpha = 0; % hyperparameter for autocorrelation matrix weighting
params.updating.beta = 0; % hyperparameter for cross-correlation vector

% cross-validation
params.cv.method = 'random'; % 'random' / 'speaker'
params.cv.nfold = 10; % number of folds in every CV repetition (testing / (training+validation))
params.cv.nrep = 5; % repetitions of CV procedure

%% Setup: parameter processing and initialization

% construct a results variable
results = struct;
results.testacc = zeros(params.cv.nrep,params.cv.nfold,length(params.subjects),length(params.windowLengths.updating),length(params.windowLengths.test),1+params.updating.iMax); % repetitions x folds x subjects x updating window lengths x test window lengths x iMax+1
results.trainacc = zeros(params.cv.nrep,params.cv.nfold,length(params.subjects),length(params.windowLengths.updating),length(params.windowLengths.test),1+params.updating.iMax); % repetitions x folds x subjects x updating window lengths x test window lengths x iMax+1

%% Loop over subjects
for sb = 1:length(params.subjects)
    fprintf('\n%s\n*** Testing subject %d ***\n%s\n',repmat('-',1,30),params.subjects(sb),repmat('-',1,30))
    
    %% data preprocessing
    % load data
    testS = params.subjects(sb);
    [eeg,envelopes,attSpeaker,fs] = loadData(params.dataset,testS,params.preprocessing);
    
    % define some extra parameters
    [nbChans,trialLength,nbTrials] = size(eeg);
    L = [floor(params.cov.L(1)*fs)+1,floor(params.cov.L(2)*fs)+1];
    
    % time-delay embedding
    nbLags = L(2)-L(1)+1;
    temp = zeros(nbLags*nbChans,trialLength,nbTrials);
    for lag = 1:nbLags
        shift = lag-1;
        temp((lag-1)*nbChans+1:lag*nbChans,1:end-shift,:) = eeg(:,1+shift:end,:);
    end
    eeg = temp;
    for ch = 1:nbChans
        eeg((ch-1)*nbLags+1:ch*nbLags,:,:) = flipud(temp(ch:nbChans:end,:,:));
    end
    clear('temp');
    
    %% cross-validation procedure
    for rep = 1:params.cv.nrep
        fprintf('\n%s\n*** Repetition nr. %d ***\n%s\n',repmat('-',1,30),rep,repmat('-',1,30))
        
        if strcmp(params.cv.method,'random')
            c{rep} = cvpartition(attSpeaker,'Kfold',params.cv.nfold);
        else
            error('Invalid cross-validation method');
        end

        % loop over CV folds
        for fold = 1:params.cv.nfold
            fprintf('\n%s\n fold nr. %d\n%s\n',repmat('-',1,15),fold,repmat('-',1,15))
            
            % -- generate a split in (training+validation)/testing data
            if strcmp(params.cv.method,'random')
                idx.train = c{rep}.training(fold);
                idx.test = c{rep}.test(fold);
            else
                idx.train = logical(1-c(:,fold));
                idx.test = logical(c(:,fold));
            end
            
            X = struct;
            X.eeg.test = eeg(:,:,idx.test);
            X.eeg.train = eeg(:,:,idx.train);
            X.envelopes.test = envelopes(:,:,idx.test);
            X.envelopes.train = envelopes(:,:,idx.train);
            labels = struct;
            labels.test = attSpeaker(idx.test);
            labels.train = attSpeaker(idx.train);
            
            %% 1. update covariance matrix and decoder
            
            % compute data covariance matrix
            switch params.cov.method
                case 'cov'
                    Rxx = cov(tens2mat(X.eeg.train,1,[])');
                case 'classic'
                    Rxx = cov(tens2mat(X.eeg.train,1,[])');
                    Rxx = Rxx + params.cov.lambda*trace(Rxx)/size(Rxx,1)*eye(size(Rxx));
                case 'lwcov'
                    Rxx = lwcov(tens2mat(X.eeg.train,1,[])');
            end
            
            % generate Rxx/rxy
            switch params.initialRxx
                case 'subject-independent'
                    % add your own
                case 'random-full'
                    RxxInit = rand(size(Rxx,1),size(Rxx,2));
            end
            
            switch params.initialRxy
                case 'subject-independent'
                    % add your own
                case 'random-full'
                    rxyInit = rand(size(Rxx,1),1);
                case 'random-y'
                    rxyInit = tens2mat(X.eeg.train,1,[])*rand(size(X.eeg.train,2)*size(X.eeg.train,3),1);
                case 'random-envelope'
                    y = [];
                    for tr = 1:size(X.envelopes.train,3)
                        env = squeeze(X.envelopes.train(randi(2),:,tr));
                        y = [y;env(:)];
                    end
                    rxyInit = tens2mat(X.eeg.train,1,[])*y;
            end
            
            % update autocorrelation matrix
            Rxx = (1-params.updating.alpha)*Rxx + params.updating.alpha*trace(Rxx)/trace(RxxInit)*RxxInit;
            
            % update decoder
            d = Rxx\rxyInit;
            
            % compute training and test accuracy
            sPred.train = squeeze(tmprod(X.eeg.train,d',1));
            sPred.test = squeeze(tmprod(X.eeg.test,d',1));
            for wTest = 1:length(params.windowLengths.test)
                pred.train = predict(sPred.train,X.envelopes.train,params.windowLengths.test(wTest)*fs);
                pred.test = predict(sPred.test,X.envelopes.test,params.windowLengths.test(wTest)*fs);
                labels.windowed.train = repelem(labels.train,floor(trialLength/(params.windowLengths.test(wTest)*fs)),1);
                labels.windowed.test = repelem(labels.test,floor(trialLength/(params.windowLengths.test(wTest)*fs)),1);
                
                results.trainacc(rep,fold,sb,:,wTest,1) = mean(labels.windowed.train==pred.train);
                results.testacc(rep,fold,sb,:,wTest,1) = mean(labels.windowed.test==pred.test);
            end
            
            %% iterative adaptation/updating
            
            rxyOld = rxyInit;
            
            % loop over windows
            for wUp = 1:length(params.windowLengths.updating)
                                            
                % iterative procedure
                for it = 1:params.updating.iMax
                    
                    %% 2. predict attended labels
                    sPred.train = squeeze(tmprod(X.eeg.train,d',1));
                    [pred.train,~,s.windowed.train] = predict(sPred.train,X.envelopes.train,params.windowLengths.updating(wUp)*fs);
                    
                    %% 3. update cross-correlation vector and decoder
                    % update cross-correlation vector
                    sAtt = [];
                    for tr = 1:size(s.windowed.train,3)
                        sAtt = [sAtt;squeeze(s.windowed.train(pred.train(tr),:,tr))'];
                    end
                    rxy = tens2mat(X.eeg.train,1,[])*sAtt;
                    rxy = (1-params.updating.beta)*rxy + params.updating.beta*norm(rxy)/norm(rxyInit)*rxyInit;
                    
                    % update decoder
                    dOld = d;
                    d = Rxx\rxy;
                    rxyOld = rxy;
                    
                    % compute training and test accuracy after updating
                    sPred.train = squeeze(tmprod(X.eeg.train,d',1));
                    sPred.test = squeeze(tmprod(X.eeg.test,d',1));
                    for wTest = 1:length(params.windowLengths.test)
                        
                        pred.train = predict(sPred.train,X.envelopes.train,params.windowLengths.test(wTest)*fs);
                        pred.test = predict(sPred.test,X.envelopes.test,params.windowLengths.test(wTest)*fs);
                        labels.windowed.train = repelem(labels.train,floor(trialLength/(params.windowLengths.test(wTest)*fs)),1);
                        labels.windowed.test = repelem(labels.test,floor(trialLength/(params.windowLengths.test(wTest)*fs)),1);
                        
                        results.trainacc(rep,fold,sb,wUp,wTest,1+it) = mean(labels.windowed.train==pred.train);
                        results.testacc(rep,fold,sb,wUp,wTest,1+it) = mean(labels.windowed.test==pred.test);
                    end

                    % stop early if convergence
                    if all(abs(dOld-d)<1e-14)
                        for wTest = 1:length(params.windowLengths.test)
                            results.trainacc(rep,fold,sb,wUp,wTest,1+it+1:end) = results.trainacc(rep,fold,sb,wUp,wTest,1+it);
                            results.testacc(rep,fold,sb,wUp,wTest,1+it+1:end) = results.testacc(rep,fold,sb,wUp,wTest,1+it);
                        end
                        break;
                    end
                end    
            end
        end
    end
    disp(squeeze(mean(mean(results.testacc(1:rep,:,1:sb,1,:,:),1),2)))
end

%% results aggregation
acc_train = squeeze(mean(mean(results.trainacc,1),2))
acc_test = squeeze(mean(mean(results.testacc,1),2))

results.params = params;
if params.save
    save(['results-',params.dataset,'-',params.saveName],'results','acc_train','acc_test','params');
end

function [predictedSpeaker,varargout] = predict(predictedEnvelope,envelopes,windowLength)

% -- segment data into windows of given length
s.predicted = segmentize(predictedEnvelope,'Segsize',windowLength);
s.predicted = reshape(s.predicted,[size(s.predicted,1),size(s.predicted,2)*size(s.predicted,3)]);

envelopes = permute(envelopes,[2,1,3]);
envelopes = segmentize(envelopes,'Segsize',windowLength);
envelopes = permute(envelopes,[3,1,2,4]);
envelopes = reshape(envelopes,[size(envelopes,1),size(envelopes,2),size(envelopes,3)*size(envelopes,4)]);
envelopes = permute(envelopes,[2,1,3]);

% -- correlate predicted envelope with recorded ones
r1 = diag(corr(s.predicted,squeeze(envelopes(:,1,:))));
r2 = diag(corr(s.predicted,squeeze(envelopes(:,2,:))));

% -- determine attended speaker
[~,predictedSpeaker] = max([r1,r2],[],2);

varargout{1} = s.predicted;
varargout{2} = permute(envelopes,[2,1,3]);
end
