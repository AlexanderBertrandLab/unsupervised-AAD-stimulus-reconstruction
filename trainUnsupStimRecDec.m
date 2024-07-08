function [d,varargout] = trainUnsupStimRecDec(X,s,RxxInit,rxsInit,updatingParams,covParams)
% UNSUPERVISEDTRAININGDECODER Perform the unsupervised training of a
% stimulus reconstruction decoder.
%
%   Input parameters:
%       X [DOUBLE, CL x T x K]: the time-lagged EEG regression matrix, with
%           K trials of length T, and C channels and L lags
%       s [DOUBLE, 2 x T x K]: the speech envelopes of the two competing
%           speakers, K trials of length T
%       RxxInit [DOUBLE, CL x CL]: the initial autocorrelation matrix
%       rxsInit [DOUBLE, CL x 1]: the initial crosscorrelation vector
%       updatingParams [STRUCT]: updating parameter variable, with fields:
%           alpha [DOUBLE]: balance between initial autocorrelation matrix
%                               and data autocorrelation matrix, between 0
%                               (no initial info) and 1 (only initial info)
%           beta [DOUBLE]: balance between initial crosscorrelation vector
%                               and updated crosscorrelation vector,
%                               between 0 (no initial info) and 1 (only
%                               initial info)
%           iMax [INTEGER]: number of updating iterations
%       covParams [STRUCT]: covariance matrix estimation parameters:
%           method [STRING]: the method to use ('lwcov' / 'ridge-reg' / 'cov') 
%           lambda [DOUBLE]: regularization parameter is method is ridge
%                               regression
%
%   Output parameters:
%       d [DOUBLE, CL x 1]: the trained decoder
%       Rxx [DOUBLE, CL x CL]: the computed autocorrelation mat. (OPTIONAL)
%       rxs [DOUBLE, CL x 1]: the computed crosscorrelation vector (OPTIONAL)

% Requires the Tensorlab toolbox for data handling
% (https://www.tensorlab.com/).

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% 1. Update autocorrelation matrix and compute initial decoder

% compute data autocorrelation matrix
switch covParams.method
    case 'cov'
        Rxx = cov(tens2mat(X,1,[])');
    case 'ridge-reg'
        Rxx = cov(tens2mat(X,1,[])');
        Rxx = Rxx + params.cov.lambda*trace(Rxx)/size(Rxx,1)*eye(size(Rxx));
    case 'lwcov'
        Rxx = lwcov(tens2mat(X,1,[])');
end

% update autocorrelation matrix
Rxx = (1-updatingParams.alpha)*Rxx + updatingParams.alpha*trace(Rxx)/trace(RxxInit)*RxxInit;

% compute initial decoder
d = Rxx\rxsInit;

%% 2. Iterative updating
for it = 1:updatingParams.iMax
    % predict labels on training set
    sRec = squeeze(tmprod(X,d',1)); % reconstruct the envelopes
    r1 = diag(corr(sRec,squeeze(s(1,:,:)))); % correlate the reconstructed envelope with speaker envelopes
    r2 = diag(corr(sRec,squeeze(s(2,:,:)))); % correlate the reconstructed envelope with speaker envelopes
    [~,predictedSpeaker] = max([r1,r2],[],2);
    
    % update cross-correlation vector
    sAtt = [];
    for w = 1:size(s,3)
        sAtt = [sAtt;squeeze(s(predictedSpeaker(w),:,w))']; % select predicted speaker
    end
    rxs = tens2mat(X,1,[])*sAtt;
    rxs = (1-updatingParams.beta)*rxs + updatingParams.beta*norm(rxs)/norm(rxsInit)*rxsInit;

    % update decoder
    d = Rxx\rxs;
end

varargout{1} = Rxx;
varargout{2} = rxs;
varargout{3} = predictedSpeaker;
end