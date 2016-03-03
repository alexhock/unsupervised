% this is demo1.m


%randn('state', 0); rand('state', 0);
%gmix = gmm(2, 4, 'spherical');
%ndat1 = 1250; ndat2 = 1250; ndat3 = 1250; ndat4 = 1250; ndata = ndat1+ndat2+ndat3+ndat4;
%gmix.centres = [2.0 0.3; 2.4 0.3; 1.5 1.1; 1.8 0.9];
%gmix.covars = [0.01 0.01 0.01 0.01];
%[data, label] = gmmsamp(gmix, ndata);

% if k is too low e.g. there are 50 clusters then it goes a bit mad, the
% std of the guassians gets really big and covers tons of the data.

% if k is high enough can still balls up

% need to set k low and high - so still need to set k


a=csvread('c:/users/ah14aeb/documents/matlab/test_data_100000_50k.csv')

data=a(:, 1:20)
labels=a(:, 21)


[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(data', 10, 60, 0, 1e-4, 0);
