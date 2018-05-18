% Compute exceedance probabilites for participants and model recovery

% 1) Participants 
%    1.1) Import csv data manually from gb_data/exceedance_probs_part.csv
%         Import as column vectors
%    1.2) Run model comparison 
%
% 
% 2) Model recovery
%    2.1) Import csv data manually from gb_data/exceedance_probs_recov.csv
%         Import as column vectors
%    2.2) Run model comparison 


% 1.2) Run model comparison 
addpath /Users/rasmus/Dropbox/spm12

lme = [e02, e1, e2, e3]
[alpha,exp_r,xp,pxp,bor] = spm_BMS(lme)

filename = '/Users/bruckner/Dropbox/gabor_bandit/code/python/gb_data/exceedance_probs_part.csv'
csvwrite(filename,xp)


% 2.2) Run model comparison 

clear all

% A0
lme_0 = [e02, e1, e2, e3]
[alpha,exp_r,xp_0,pxp,bor] = spm_BMS(lme_0)

% A1
lme_1 = [e4, e5, e6, e7]
[alpha,exp_r,xp_1,pxp,bor] = spm_BMS(lme_1)

% A2
lme_2 = [e8, e9, e10, e11]
[alpha,exp_r,xp_2,pxp,bor] = spm_BMS(lme_2)

% A3
lme_3 = [e12, e13, e14, e15]
[alpha,exp_r,xp_3,pxp,bor] = spm_BMS(lme_3)

filename = '/Users/bruckner/Dropbox/gabor_bandit/code/python/gb_data/exceedance_probs_recov.csv'
M = [xp_0; xp_1;xp_2; xp_3]
csvwrite(filename,M)

