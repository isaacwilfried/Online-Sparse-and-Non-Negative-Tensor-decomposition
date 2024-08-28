clc;
clear;
clear all;
close all;

% Load the data from the .mat file
load('tenseur_4composants_corrige_zepp_02_11_2020.mat'); 

% Extract the tensor at time t0
X1 = Xf(:,:,1:10);

% Set the rank (overestimated)
R = 7;
% Call the SNNCPD_gradientNadam function to compute factors
[A_ntda, B_ntda, Cre, Da, Db, Va, Vb, hist, MU] = SNNCPD_gradientNadam(X1, R);

% seach best parameters (take time)
%%[A_ntda, B_ntda, Cre, best_params, best_hist] = optimize_SNNCPD(X1, R);

% Plot the results
newFig = figure;
subplot(2,2,1);
plot(A_ntda);
title('Factor A');

subplot(2,2,2);
plot(B_ntda);
title('Factor B');

subplot(2,2,3);
plot(Cre);
title('Factor Cre');

% Custom function to display components (adjust as needed)
displaycomponent(A_ntda, B_ntda, R);


% Define legends (adjust strings as per the data context)
legend_strE = {'5S8HQ','Tryptophane','Fluorosceine','Rhodamine'};
legend_strEm = {'Component 1-Emission','Component 2-Emission','Component 3-Emission','Component 4-Emission'};




