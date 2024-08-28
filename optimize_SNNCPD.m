function [best_A, best_B, best_C, best_params, best_hist] = optimize_SNNCPD(T, R, param_grid)
    % OPTIMIZE_SNNCPD Performs grid search to optimize the hyperparameters for SNNCPD
    %
    % INPUTS:
    %   T         - The input tensor (3rd order)
    %   R         - The rank for tensor decomposition
    %   param_grid - Structure containing the grid of parameters to search
    %                .step  - Array of learning rates to test
    %                .beta1 - Array of beta1 values to test (momentum for NADAM)
    %                .beta2 - Array of beta2 values to test (momentum for NADAM)
    %                .alpha - Array of alpha values to test (L1 regularization coefficient)
    %
    % OUTPUTS:
    %   best_A, best_B, best_C - Factor matrices corresponding to the best parameters found
    %   best_params            - Structure containing the best parameters found
    %   best_hist              - History of the relative reconstruction error for the best parameters
    
    % Set default parameter grid if none provided
    if nargin < 3
        param_grid.step = [1e-1, 1e-2, 1e-3];
        param_grid.beta1 = [0.9, 0.95];
        param_grid.beta2 = [0.9, 0.99];
        param_grid.alpha = [0.1, 0.5, 1.0];
    end
    
    % Initialize variables to store the best results
    best_cost = Inf;
    best_A = [];
    best_B = [];
    best_C = [];
    best_params = struct();
    best_hist = [];

    % Number of iterations to consider for averaging the error
    n_avg = 10;

    % Loop through all combinations of parameters in the grid
    for step = param_grid.step
        for beta1 = param_grid.beta1
            for beta2 = param_grid.beta2
                for alpha = param_grid.alpha
                    % Create options structure for the current parameter combination
                    options = createOptions(1, 1, alpha, beta1, beta2,step);
                    
                    % Run the optimization with the current parameters
                    [A, B, C, Da, Db, Va, Vb, hist, MU] = SNNCPD_gradientNadam(T, R, options, step);
                    
                    % Calculate the average of the last n_avg error values
                    if length(hist) >= n_avg
                        avg_cost = mean(hist(end-n_avg+1:end));
                    else
                        avg_cost = mean(hist);
                    end
                    
                    % Check if this is the best combination so far
                    if avg_cost < best_cost
                        best_cost = avg_cost;
                        best_A = A;
                        best_B = B;
                        best_C = C;
                        best_params.step = step;
                        best_params.beta1 = beta1;
                        best_params.beta2 = beta2;
                        best_params.alpha = alpha;
                        best_hist = hist;
                    end
                end
            end
        end
    end
    
    % Display the best parameters found
    fprintf('Best Parameters Found:\n');
    fprintf('Step: %f\n', best_params.step);
    fprintf('Beta1: %f\n', best_params.beta1);
    fprintf('Beta2: %f\n', best_params.beta2);
    fprintf('Alpha: %f\n', best_params.alpha);
end


function [A,B,C,Da,Db,Va,Vb,hist,MU] = SNNCPD_gradientNadam(T,R,options,step)
%% SNNCPD = Sparse NonNegative CPD
% This function performs CPD of a 3rd order tensor using dictionary learning
% and the NADAM optimizer (a variant of stochastic gradient descent). The function allows
% for overestimation of the rank when the true rank is unknown due to the sparsity 
% imposed on the cost function. Non-negativity constraints are applied for positive data.

% Cost Function:
% 0.5*|| T - DA*VA \circ DB*VB \circ C||_F^2 + alpha||Va||_{1,1} + alpha||Vb||_{1,1}

% 
% ----------------------INPUT---------------------
% T : 3rd order tensor
% R : Rank 

% ----------------OPTIONAL INPUT---------------------
% options : Optional parameters
%   case 0 : No sparsity. In this case, R is known.
%   case 1 : L1 norm on VA, VB. In this case, R is unknown and overestimated (default).
%   alpha : Penalty coefficient
%   nonnegativity 0 : No non-negativity constraint
%   nonnegativity 1 : With non-negativity constraint
%   beta1 : Chosen between [0.1;0.9], default beta1 = 0.9
%   beta2 : Chosen between [0.1;0.9], default beta2 = 0.9
%   step  : Learning rate for the optimization process.
             
% ----------------------OUTPUT---------------------
% Da, Db : Dictionaries
% Va, Vb : Weights or atoms
% A : Factor matrix (A=Da*Va)
% B : Factor matrix (B=Db*Vb)
% C : Factor matrix
% hist : Relative reconstruction error for each iteration
% MU : Learning rate for each iteration

% ----------------------EXAMPLES---------------------
% [A,B,C,Da,Db,Va,Vb,hist,MU] = SNNCPD_gradientNadam(T,R);
% [A,B,C,Da,Db,Va,Vb,hist,MU] = SNNCPD_gradientNadam(T,R,[]);
% [A,B,C,Da,Db,Va,Vb,hist,MU] = SNNCPD_gradientNadam(T,R,creerOptions(1, 1, 1.5, 0.6,0.7));

% ----------------------USED FUNCTIONS---------------------
% C = renormaliser_dictionnaire2(C,norma) Normalize dictionary
% [X,norma] = normalisation_tenseur2(T) Normalize tensor


% ----------------------REFs---------------------
% [1]saac Wilfried Sanou, Roland Redon, Xavier Luciani, and Stephane Mounier. Online nonne-
% gative and sparse canonical polyadic decomposition of fluorescence tensors. Chemometrics
% and Intelligent Laboratory Systems, 225 :104550, 2022.
% [2] Timothy Dozat. Incorporating NESTEROV momentum into ADAM. Proceedings of 4th
% International Conference on Learning Representations, Workshop Track, 2016.

%
% Auteur : ISAAC SANOU
% Date de creation : 01/02/2020
% Date de modification : 11/07/2022 

tic
%% Default parameters
if(nargin < 3 || isempty(options))
    options = createOptions(1, 1, 2, 0.9, 0.9, 1e-3);
end

% Use the step size from options
step = options.step;

% Normalization of the tensor
[T,norma] = normalisation_tenseur2(T);

% Unfolding the tensor
T1 = unfold(T, [1 2 3]);
T2 = unfold(T, [2 3 1]);
T3 = unfold(T, [3 1 2]);

% Max iterations
maxit = 20000;
itTot = 0;

% Stopping criterion
crit = 10^-16;
decay = 1e-6; % NADAM parameter

%% Initialization 
Da = max(0, randn(size(T,1), R));
Db = max(0, randn(size(T,2), R));
C  = max(0, randn(size(T,3), R));

Va = eye(R, R);
Vb = eye(R, R);

l1_va = ones(size(Va));
l1_vb = ones(size(Vb));

% NADAM parameters (momentum 1 & 2)
M_va = zeros(R, R);
M_vb = zeros(R, R);
M_c = zeros(size(T,3), R);
M_da = zeros(size(T,1), R);
M_db = zeros(size(T,2), R);

Mm_va = zeros(R, R);
Mm_vb = zeros(R, R);
Mm_c = zeros(size(T,3), R);
Mm_da = zeros(size(T,1), R);
Mm_db = zeros(size(T,2), R);

% Objective function
XDep = T1;
SSX = sum(sum(XDep.^2));
obj0 = inf;

%% Choosing sparsity or not
if options.cas == 0
    step = 1e-1; % Override step if no sparsity
end

if options.cas == 1
    step = 1e-3; % Override step if sparsity is enforced
end

relerr1 = 1;

while (relerr1> crit && itTot < maxit)
    itTot = itTot + 1;
    
    % Compute gradient
    [G_da,G_db,G_c,G_va,G_vb] = compute_gradient(T1,T2,T3,Da,Db,Va,Vb,C,l1_va,l1_vb,options.cas,options.alpha);
    
    % Update NADAM parameters
    M_va = options.beta1*M_va + (1 - options.beta1) * G_va;
    M_vb = options.beta1*M_vb + (1 - options.beta1) * G_vb;
    M_c = options.beta1*M_c + (1 - options.beta1) * G_c;
    M_da = options.beta1*M_da + (1 - options.beta1) * G_da;
    M_db = options.beta1*M_db + (1 - options.beta1) * G_db;
    
    Mm_va = options.beta2*Mm_va + (1 - options.beta2) * G_va.^2;
    Mm_vb = options.beta2*Mm_vb + (1 - options.beta2) * G_vb.^2;
    Mm_c = options.beta2*Mm_c + (1 - options.beta2) * G_c.^2;
    Mm_da = options.beta2*Mm_da + (1 - options.beta2) * G_da.^2;
    Mm_db = options.beta2*Mm_db + (1 - options.beta2) * G_db.^2;
    
    beta1_t = options.beta1.^itTot;
    mbeta1 = 1 - beta1_t;
    M_va1 = M_va./mbeta1;
    M_vb1 = M_vb ./mbeta1;
    M_c1 = M_c ./ mbeta1;
    M_da1 = M_da ./mbeta1;
    M_db1 = M_db ./ mbeta1;
    
    beta2_t = options.beta2.^itTot;
    mbeta2 = 1 - beta2_t;
    
    Mm_va1 = Mm_va./mbeta2;
    Mm_vb1 = Mm_vb ./mbeta2;
    Mm_c1 = Mm_c ./ mbeta2;
    Mm_da1 = Mm_da ./mbeta2;
    Mm_db1 = Mm_db ./ mbeta2;
    
    %% Update factor matrices
    if options.nonnegativite == 1 % With non-negativity constraint
        Da1 = max(Da - step./(sqrt(Mm_da1)+decay) .* (options.beta1*M_da1 + (1 - options.beta1/1-beta1_t) .* G_da), 0);
        Db1 = max(Db - step./(sqrt(Mm_db1)+decay) .* (options.beta1*M_db1 + (1 - options.beta1/1-beta1_t) .* G_db), 0);
        C1 = max(C - step./(sqrt(Mm_c1)+decay) .* (options.beta1*M_c1 + (1 - options.beta1/1-beta1_t) .* G_c), 0);
        Va1 = max(Va - step./(sqrt(Mm_va1)+decay) .* (options.beta1*M_va1 + (1 - options.beta1/1-beta1_t) .* G_va), 0);
        Vb1 = max(Vb - step./(sqrt(Mm_vb1)+decay) .* (options.beta1*M_vb1 + (1 - options.beta1/1-beta1_t) .* G_vb), 0);
    else % Without non-negativity constraint
        Da1 = Da - step./(sqrt(Mm_da1)+decay) .* (options.beta1*M_da1 + (1 - options.beta1/1-beta1_t) .* G_da);
        Db1 = Db - step./(sqrt(Mm_db1)+decay) .* (options.beta1*M_db1 + (1 - options.beta1/1-beta1_t) .* G_db);
        C1 = C - step./(sqrt(Mm_c1)+decay) .* (options.beta1*M_c1 + (1 - options.beta1/1-beta1_t) .* G_c);
        Va1 = Va - step./(sqrt(Mm_va1)+decay) .* (options.beta1*M_va1 + (1 - options.beta1/1-beta1_t) .* G_va);
        Vb1 = Vb - step./(sqrt(Mm_vb1)+decay) .* (options.beta1*M_vb1 + (1 - options.beta1/1-beta1_t) .* G_vb);
    end
    
    %% Compute cost function
    obj = calculerErreur(XDep, Da1*Va1, Db1*Vb1, C1, Va1, Vb1, options.cas, options.alpha);
    relerr1 = abs(obj-obj0)/(obj);
    hist(itTot) = relerr1;

    %% Stopping criterion
    if obj/SSX < 1000*eps % Getting close to the machine uncertainty => stop
        relerr1 = 0;
        break;
    end
     
    %% Adjust learning rate
    val_ref = 0.05; % Increase step if f_new < f_old
    if obj > obj0
        step = step/2;
    elseif relerr1 < val_ref
        step = 1.2*step;
    end
    
    Da = Da1;
    Db = Db1;
    C = C1;
    Va = Va1;
    Vb = Vb1;
    obj0 = obj;
    
    MU(itTot) = step;    
end

% Renormalization
C = renormaliser_dictionnaire2(C,norma);

% Recover the correct rank
[A,B,C] = best_rank(Da,Db,C,Va,Vb);

% Sort columns
[A,B,C] = triModes(A,B,C,2);

toc
end

%% Normalization
function [X,norma] = normalisation_tenseur2(T)
    taille = size(T,3);
    X = T;
    norma = zeros(1,taille);

    for k = 1:taille
        ff = T(:,:,k);
        norma(k) = max(abs(ff(:)));
        X(:,:,k) = T(:,:,k)./norma(k);
    end
end

%% Gradient clipping default = none
function [newgradient] = clipping(gradient,theta)
    newgradient = gradient * theta / norm(gradient,2);
end

%% Choosing the best rank
function [A,B,C] = best_rank(Da,Db,Ce,Va,Vb)
    [row,col] = find(sum(Va)==0);
    tf = isempty(col);
    if tf == 1
        A = Da*Va;
        B = Db*Vb;
        C = Ce;
    elseif tf == 0
        A = Da*Va;
        B =Db*Vb;
        C = Ce;
        A(:,col) = 0;
        B(:,col) = 0;
        C(:,col) = 0;
    end
end

%% Sorting the modes
function [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)
% Fonction triModes
%       /global
%
% Examples :
%       [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)
%       [Modestries] = triModes(Modes,num)
%
% Sorting the columns of modes A, B, and C (or the array of modes)
% based on the increasing position of the maximum of mode number `num`.
%
% Inputs:
%       A,B,C : initial matrices (from the Factors table of a PARAFAC)
%       num   : 1, 2, or 3
%
% Outputs:
%       Atrie,Btrie,Ctrie : sorted modes based on the position of the maximum

    if ( (nargin==4) && (nargout==3) )
        switch num
            case 1
                [~,posmax] = max(A);
            case 2
                [~,posmax] = max(B);
            case 3
                [~,posmax] = max(C);
        end
        [~,indtri] = sort(posmax);
        Atrie = A(:,indtri);
        Btrie = B(:,indtri);
        Ctrie = C(:,indtri);
    end

    if ( (nargin==2) && (nargout==1) && iscell(A) )
        num = B;
        [~,posmax] = max(A{num});
        [~,indtri] = sort(posmax);
    
        nA = length(A);
        Atrie = cell(1,nA);
        for k = 1:nA
            Atrie{k} = A{k}(:,indtri);
        end
    end
end

%% Renormalizing the dictionary
function C = renormaliser_dictionnaire2(Ctmp,norma)
    C = Ctmp .* repmat(norma(:), [1 size(Ctmp,2)]);
end
