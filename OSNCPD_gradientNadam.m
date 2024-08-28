function [A,B,C,Ua,Ub,Da,Db,Va,Vb,hist,MU,grad,obj,relative_error] = OSNCPD_gradientNadam(T,R,cas,alpha,beta1,beta2,Da,Db,va,vb)
%% NTDA = Online Nonnegative Tensor Dictionary Approximation

% This function performs an online nonnegative tensor dictionary approximation
% with the NADAM optimizer. It includes an option for enforcing orthogonality
% constraints on the transformation matrices Ua and Ub.

% ----------------------INPUT---------------------
% T      : 3rd-order tensor
% R      : Rank of the decomposition
% cas    : Type of regularization
%          0 -> No penalty
%          1 -> L1 norm on Va, Vb
% alpha  : Regularization parameter for sparsity
% beta1  : First momentum parameter for NADAM
% beta2  : Second momentum parameter for NADAM
% Da, Db : Known dictionaries
% va, vb : Initial weights or atoms

% ----------------------OUTPUT---------------------
% A, B, C        : Factor matrices
% Ua, Ub         : Orthogonal transformation matrices
% Da, Db         : Updated dictionaries
% Va, Vb         : Updated weights or atoms
% hist           : History of the objective function value
% MU             : Learning rate history
% grad           : Gradient norm history
% obj            : Final objective function value
% relative_error : Relative error at each iteration

%% Initialization
tol = 1e-16; % Tolerance for convergence
itTot = 0; % Total iterations
nouvelleErr = 1000; % Initial error
maxit = 10000; % Maximum iterations
clippingG = 0; % Gradient clipping flag
theta = 1; % Clipping parameter
obj0 = 1; % Initial objective value
SSX = 1000; % Initial squared error sum
decay = 1e-2; % Decay rate for learning rate

% Normalize the tensor
[T, norma] = normalisation_tenseur2(T);

% Unfold the tensor along different modes
T1 = deplier(T, [1 2 3]);
T2 = deplier(T, [2 3 1]);
T3 = deplier(T, [3 1 2]);
Mnrm = norm(T1);

% Initialize dictionaries and add noise
Da = Da * va;
Db = Db * vb;

% Add noise to zero columns in Da and Db
for i = 1:R
    if sum(Da(:,i)) == 0
        Da(:,i) = randn(size(T,1), 1);
        Db(:,i) = randn(size(T,2), 1);
    end
end

% Initialize transformation matrices as identity matrices
Ua = eye(size(T,1), size(T,1));
Ub = eye(size(T,2), size(T,2));

% Initialize weights and factor matrices
Va = eye(R,R);
Vb = eye(R,R);
C  = max(0, eye(size(T,3),R)); % Initial factor matrix

% Initialize gradient and second moment matrices for NADAM
M_va = zeros(R,R);
M_vb = zeros(R,R);
M_c = zeros(size(T,3),R);
M_Ua = zeros(size(T,1),size(T,1));
M_Ub = zeros(size(T,2),size(T,2));

Mm_va = zeros(R,R);
Mm_vb = zeros(R,R);
Mm_c = zeros(size(T,3),R);
Mm_Ua = zeros(size(T,1),size(T,1));
Mm_Ub = zeros(size(T,2),size(T,2));

% L1 norm derivatives
l1_va = ones(size(Va));
l1_vb = ones(size(Vb));

%% Learning rate selection based on regularization type
if cas == 0
    step = 1e-1; % Learning rate without sparsity
elseif cas == 1
    step = 1e-9; % Learning rate with L1 sparsity
end

%% Main Optimization Loop
while itTot < maxit
    itTot = itTot + 1; % Increment iteration counter
    
    % Compute the gradients with respect to all variables
    [G_Ua, G_Ub, G_c, G_va, G_vb] = compute_gradientOnline(T1, T2, T3, Da, Ua, Ub, Db, Va, Vb, C, l1_va, l1_vb, cas, alpha);
    
    % Optional gradient clipping (if enabled)
    if clippingG == 1
        G_Ua = clipping(G_Ua, theta);
        G_va = clipping(G_va, theta);
        G_Ub = clipping(G_Ub, theta);
        G_c = clipping(G_c, theta);
        G_vb = clipping(G_vb, theta);
    end
    
    % NADAM parameter updates
    M_va = beta1 * M_va + (1 - beta1) * G_va;
    M_vb = beta1 * M_vb + (1 - beta1) * G_vb;
    M_c = beta1 * M_c + (1 - beta1) * G_c;
    M_Ua = beta1 * M_Ua + (1 - beta1) * G_Ua;
    M_Ub = beta1 * M_Ub + (1 - beta1) * G_Ub;
    
    Mm_va = beta2 * Mm_va + (1 - beta2) * G_va.^2;
    Mm_vb = beta2 * Mm_vb + (1 - beta2) * G_vb.^2;
    Mm_c = beta2 * Mm_c + (1 - beta2) * G_c.^2;
    Mm_Ua = beta2 * Mm_Ua + (1 - beta2) * G_Ua.^2;
    Mm_Ub = beta2 * Mm_Ub + (1 - beta2) * G_Ub.^2;
    
    % Corrected first and second moments for bias
    beta1_t = beta1.^itTot;
    mbeta1 = 1 - beta1_t;
    M_va1 = M_va ./ mbeta1;
    M_vb1 = M_vb ./ mbeta1;
    M_c1 = M_c ./ mbeta1;
    M_Ua1 = M_Ua ./ mbeta1;
    M_Ub1 = M_Ub ./ mbeta1;
    
    beta2_t = beta2.^itTot;
    mbeta2 = 1 - beta2_t;
    Mm_va1 = Mm_va ./ mbeta2;
    Mm_vb1 = Mm_vb ./ mbeta2;
    Mm_c1 = Mm_c ./ mbeta2;
    Mm_Ua1 = Mm_Ua ./ mbeta2;
    Mm_Ub1 = Mm_Ub ./ mbeta2;
    
    %% Update factor matrices with orthogonality constraint
    Ua1 = Ua - step ./ (sqrt(Mm_Ua1) + 1e-9) .* (beta1 * M_Ua1 + (1 - beta1 / mbeta1) .* G_Ua);
    Ub1 = Ub - step ./ (sqrt(Mm_Ub1) + 1e-9) .* (beta1 * M_Ub1 + (1 - beta1 / mbeta1) .* G_Ub);
    
    % Enforce orthogonality using QR decomposition
    [Ua1, ~] = qr(Ua1, 0); % Orthogonalize Ua1
    [Ub1, ~] = qr(Ub1, 0); % Orthogonalize Ub1
    
    % Update other factor matrices
    Va1 = max(Va - step ./ (sqrt(Mm_va1) + 1e-9) .* (beta1 * M_va1 + (1 - beta1 / mbeta1) .* G_va), 0);
    Vb1 = max(Vb - step ./ (sqrt(Mm_vb1) + 1e-9) .* (beta1 * M_vb1 + (1 - beta1 / mbeta1) .* G_vb), 0);
    C1 = max(C - step ./ (sqrt(Mm_c1) + 1e-9) .* (beta1 * M_c1 + (1 - beta1 / mbeta1) .* G_c), 0);
    
    %% Compute objective function and error
    obj = calculerErreurOnline(T1, Da, Db, Ua1, Ub1, C1, Va1, Vb1, cas, alpha);
    hist(itTot) = obj;
    relerr1 = abs(obj - obj0) / obj0;
    relerr2 = sqrt(2 * obj) / Mnrm;
    
    %% Adjust step size based on error
    val_ref = 0.005; % Threshold for adjusting learning rate
    if obj > obj0
        step = step / 2;
    elseif relerr1 < val_ref
        step = 1.2 * step;
    end
    
    % Update variables for the next iteration
    Ua = Ua1;
    Ub = Ub1;
    C = C1;
    Va = Va1;
    Vb = Vb1;
    obj0 = obj;
    
    % Check for convergence
    crit = relerr1 < tol;
    if crit
        nstall = nstall + 1;
    else
        nstall = 0;
    end
    if nstall >= 100 || relerr2 < tol
        break;
    end
    
    % Store relative error and gradient norm for analysis
    relative_error(itTot) = relerr1;
    grad(itTot) = norm([Ua(:); Va1(:); Ub(:); Vb1(:); C1(:)]);
end

% Renormalize the dictionary C
C = renormaliser_dictionnaire2(C, norma);

% Recover the correct rank and sort components
[A, B, C] = best_rank(Ua * Da, Ub * Db, C, Va, Vb);
[A, B, C] = triModes(A, B, C, 2);

end

%% Normalization function for the tensor
function [X, norma] = normalisation_tenseur2(T)
    taille = size(T, 3);
    X = T;
    norma = zeros(1, taille);
    for k = 1:taille
        ff = T(:,:,k);
        norma(k) = max(abs(ff(:)));
        X(:,:,k) = T(:,:,k) ./ norma(k);
    end
end

%% Gradient clipping function (default = none)
function [newgradient] = clipping(gradient, theta)
    newgradient = gradient * theta / norm(gradient, 2);
end

%% Function to find the best rank
function [A, B, C] = best_rank(Da, Db, Ce, Va, Vb)
    [row, col] = find(sum(Va) == 0);
    if isempty(col)
        A = Da * Va;
        B = Db * Vb;
        C = Ce;
    else
        A = Da * Va;
        B = Db * Vb;
        C = Ce;
        A(:, col) = 0;
        B(:, col) = 0;
        C(:, col) = 0;
    end
end

%% Function to sort the modes based on the position of the maximum
function [Atrie, Btrie, Ctrie] = triModes(A, B, C, num)
    switch num
        case 1
            [~, posmax] = max(A);
        case 2
            [~, posmax] = max(B);
        case 3
            [~, posmax] = max(C);
    end
    [~, indtri] = sort(posmax);
    Atrie = A(:, indtri);
    Btrie = B(:, indtri);
    Ctrie = C(:, indtri);
end

%% Function to renormalize the dictionary
function C = renormaliser_dictionnaire2(Ctmp, norma)
    C = Ctmp .* repmat(norma(:), [1 size(Ctmp, 2)]);
end