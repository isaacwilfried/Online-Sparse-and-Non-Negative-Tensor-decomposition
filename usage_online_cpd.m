% Main Script for Tensor Decomposition
% Description: This script demonstrates the use of SNNCPD for initialization
% and subsequently applies online algorithms (OSNNCPD1, OSNNCPD2, NTDA) to
% process new tensor slices as they arrive.
clf
clear all

%% 1. Initialization using SNNCPD
% Load or create your initial tensor T0
T0 = rand(10, 10, 10); % Example initial tensor

% Set the rank for the decomposition
R = 5;

% Initialize with SNNCPD
[A0, B0, C0, Da0, Db0, Va0, Vb0, hist_SNNCPD, MU_SNNCPD] = SNNCPD_gradientNadam(T0, R);

% Display initialization results
disp('SNNCPD Initialization Complete');
disp('Initial Factors:');
disp(A0); disp(B0); disp(C0);

%% 2. Online Update using OSNNCPD1
% Assume T1 is the next slice of the tensor arriving online
T1 = rand(10, 10, 1); % Example next slice

% Update using OSNNCPD1
[A1, B1, C1, Da1, Db1, Va1, Vb1, hist_OSNNCPD1, MU_OSNNCPD1] = OSNNCPD1_gradientNadam(T1, R, Da0, Db0, Va0);

% Display online update results for OSNNCPD1
disp('OSNNCPD1 Update Complete');
disp('Updated Factors:');
disp(A1); disp(B1); disp(C1);

%% 3. Online Update using OSNNCPD2
% Assume T2 is another slice arriving online
T2 = rand(10, 10, 1); % Example next slice

% Update using OSNNCPD2
[A2, B2, C2, Ua2, Ub2, Da2, Db2, Va2, Vb2, hist_OSNNCPD2, MU_OSNNCPD2] = OSNNCPD2_gradientNadam(T2, R, Da0, Db0, Va0, Vb0);

% Display online update results for OSNNCPD2
disp('OSNNCPD2 Update Complete');
disp('Updated Factors:');
disp(A2); disp(B2); disp(C2);

%% 4. Online Update using NTDA with Orthogonality Constraint
% Assume T3 is another slice arriving online
T3 = rand(10, 10, 1); % Example next slice

% Update using NTDA with orthogonality constraints
[A3, B3, C3, Ua3, Ub3, Da3, Db3, Va3, Vb3, hist_NTDA, grad_NTDA, obj_NTDA, relative_error_NTDA] = ...
    OOSCPD_gradientNadam(T3, R, 1, 0.1, 0.9, 0.999, Da0, Db0, Va0, Vb0);

% Display online update results for NTDA
disp('NTDA Update Complete');
disp('Updated Factors with Orthogonality:');
disp(A3); disp(B3); disp(C3);

%% Final Remarks
% The script demonstrates the process of initializing a tensor decomposition
% using SNNCPD and updating the decomposition online as new data slices arrive.
% This approach is applicable in real-time data analysis scenarios where tensor data
% streams in continuously, such as in signal processing or video analysis.

