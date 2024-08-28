function [A,B,C,Da,Db,Va,Vb,hist,MU] = OSNNCPD1_gradientNadam(T,R,Dat0,Dbt0,Vat0,options)
%% OSNNCPD1 = Online and Sparse NonNegative CPD 1
% This function performs an online CPD of a 3rd order tensor using dictionary
% learning with the NADAM optimizer (a variant of stochastic gradient descent).
% The function allows for overestimation of the rank when the true rank is unknown
% due to the sparsity imposed on the cost function. Non-negativity constraints
% are applied for positive data.
%
% Cost Function:
% 0.5*|| T - DA*VA \circ DB*VB \circ C||_F^2 + alpha||Va||_{1,1}+ alpha||Vb||_{1,1}
%
% ----------------------INPUT---------------------
% T      : 3rd order tensor
% R      : Rank
% Dat0, Dbt0 : Dictionaries obtained in the previous time step using SNNCPD
% Vat0   : Atoms obtained in the previous time step
%
% ----------------OPTIONAL INPUT---------------------
% options: Optional parameters
%   cas          : 0 -> No sparsity. R is known.
%                  1 -> L1 norm on VA, VB. R is unknown and overestimated (default).
%   alpha        : Penalty coefficient
%   nonnegativite: 0 -> No non-negativity constraint
%                  1 -> With non-negativity constraint
%   beta1        : Momentum parameter for NADAM, chosen between [0.1;0.9], default = 0.9
%   beta2        : Second momentum parameter for NADAM, chosen between [0.1;0.9], default = 0.9
%
% ----------------------OUTPUT---------------------
% Da, Db : Dictionaries
% Va, Vb : Weights or atoms
% A, B, C : Factor matrices
% hist  : Relative reconstruction error for each iteration
% MU    : Learning rate for each iteration
%
% ----------------------EXAMPLES---------------------
% [A,B,C,Da,Db,Va,Vb,hist,MU] = OSNNCPD1_gradientNadam(T,R,Dat0,Dbt0,Vat0);
% [A,B,C,Da,Db,Va,Vb,hist,MU] = OSNNCPD1_gradientNadam(T,R,Dat0,Dbt0,Vat0,[]);
% [A,B,C,Da,Db,Va,Vb,hist,MU] = OSNNCPD1_gradientNadam(T,R,Dat0,Dbt0,Vat0,creerOptions(1, 1, 1.5, 0.6, 0.7));

% ----------------------USED FUNCTIONS---------------------
% C = renormaliser_dictionnaire2(C,norma) Normalize dictionary
% [X,norma] = normalisation_tenseur2(T) Normalize tensor
% [A,B,C] = best_rank(Da,Db,C,Va,Vb) Find the correct rank
% [A,B,C] = triModes(A,B,C,2) Sort components in ascending order

% ----------------------REFs---------------------
% [1] Isaac Wilfried Sanou, Roland Redon, Xavier Luciani, and Stéphane Mounier. Online non-negative
% and sparse canonical polyadic decomposition of fluorescence tensors. Chemometrics
% and Intelligent Laboratory Systems, 225:104550, 2022.
% [2] Timothy Dozat. Incorporating NESTEROV momentum into ADAM. Proceedings of 4th
% International Conference on Learning Representations, Workshop Track, 2016.

% Author: ISAAC SANOU
% Creation Date: 01/02/2020
% Modification Date: 11/07/2022

%% Default parameters
if nargin < 6 || isempty(options)
    options = createOptions(1, 1, 5, 0.9, 0.9, 1e-3);
end

%% Identify indices from previous atoms Vat0
ind= [];
for i=1:size(Vat0,1)
    for j=1:size(Vat0,2)
        if Vat0(i,j) > 0
            ind = [ind,i];
        end
    end
end

% Normalize the tensor
[T, norma] = normalisation_tenseur2(T);

% Unfold the tensor
T1 = deplier(T, [1 2 3]);
T2 = deplier(T, [2 3 1]);
T3 = deplier(T, [3 1 2]);

% Set parameters
maxit = 10000;
itTot = 0;
crit = 1e-16;
relerr1 = 1;
decay = 1e-6;

%% Initialization
Da = Dat0;
Db = Dbt0;
C  = max(0, zeros(size(T,3), R));  % Ensure non-negativity
Va = eye(R, R);
Vb = eye(R, R);

% NADAM parameters (momentum 1 & 2)
M_va=zeros(R,R);
M_vb =zeros(R,R);
M_c =zeros(size(T,3),R);
M_da =zeros(size(T,1),R);
M_db = zeros(size(T,2),R);

Mm_va=zeros(R,R);
Mm_vb =zeros(R,R);
Mm_c =zeros(size(T,3),R);
Mm_da =zeros(size(T,1),R);
Mm_db = zeros(size(T,2),R);

% L1 norm derivative
l1_va = ones(size(Va));
l1_vb = ones(size(Vb));

% Objective function initialization
XDep = deplier(T, [1 2 3]);
SSX = sum(sum(XDep.^2));
obj0 = inf;

%% Sparsity constraint selection
if options.cas == 0
    step = 1e-1; % Learning rate for no sparsity
else
    step = 1e-3; % Learning rate for sparsity
end

while (relerr1 > crit && itTot < maxit)
    itTot = itTot + 1;
    % Compute gradient
    [G_da, G_db, G_c, G_va, G_vb] = compute_gradient(T1, T2, T3, Da, Db, Va, Vb, C, l1_va, l1_vb, options.cas, options.alpha);

    
    % Zero-out gradients for the indices corresponding to the previous atoms
    G_da(:, ind) = 0;
    G_db(:, ind) = 0;
    

    % NADAM parameter update
    M_va = options.beta1 * M_va + (1 - options.beta1) * G_va;
    M_vb = options.beta1 * M_vb + (1 - options.beta1) * G_vb;
    M_c = options.beta1 * M_c + (1 - options.beta1) * G_c;

    M_da = options.beta1 * M_da + (1 - options.beta1) * G_da;
    M_db = options.beta1 * M_db + (1 - options.beta1) * G_db;

    Mm_va = options.beta2 * Mm_va + (1 - options.beta2) * G_va.^2;
    Mm_vb = options.beta2 * Mm_vb + (1 - options.beta2) * G_vb.^2;
    Mm_c = options.beta2 * Mm_c + (1 - options.beta2) * G_c.^2;
    Mm_da = options.beta2 * Mm_da + (1 - options.beta2) * G_da.^2;
    Mm_db = options.beta2 * Mm_db + (1 - options.beta2) * G_db.^2;

    beta1_t = options.beta1.^itTot;
    mbeta1 = 1 - beta1_t;
    M_va1 = M_va ./ mbeta1;
    M_vb1 = M_vb ./ mbeta1;
    M_c1 = M_c ./ mbeta1;
    M_da1 = M_da ./ mbeta1;
    M_db1 = M_db ./ mbeta1;

    beta2_t = options.beta2.^itTot;
    mbeta2 = 1 - beta2_t;

    Mm_va1 = Mm_va ./ mbeta2;
    Mm_vb1 = Mm_vb ./ mbeta2;
    Mm_c1 = Mm_c ./ mbeta2;
    Mm_da1 = Mm_da ./ mbeta2;
    Mm_db1 = Mm_db ./ mbeta2;

    %% Update factor matrices
    if options.nonnegativite == 1 % With non-negativity constraint
        Da1 = max(Da - step ./ (sqrt(Mm_da1) + decay) .* (options.beta1 * M_da1 + (1 - options.beta1) ./ mbeta1 .* G_da), 0);
        Db1 = max(Db - step ./ (sqrt(Mm_db1) + decay) .* (options.beta1 * M_db1 + (1 - options.beta1) ./ mbeta1 .* G_db), 0);
        C1 = max(C - step ./ (sqrt(Mm_c1) + decay) .* (options.beta1 * M_c1 + (1 - options.beta1) ./ mbeta1 .* G_c), 0);
        Va1 = max(Va - step ./ (sqrt(Mm_va1) + decay) .* (options.beta1 * M_va1 + (1 - options.beta1) ./ mbeta1 .* G_va), 0);
        Vb1 = max(Vb - step ./ (sqrt(Mm_vb1) + decay) .* (options.beta1 * M_vb1 + (1 - options.beta1) ./ mbeta1 .* G_vb), 0);
    else % Without non-negativity constraint
        Da1 = Da - step ./ (sqrt(Mm_da1) + decay) .* (options.beta1 * M_da1 + (1 - options.beta1) ./ mbeta1 .* G_da);
        Db1 = Db - step ./ (sqrt(Mm_db1) + decay) .* (options.beta1 * M_db1 + (1 - options.beta1) ./ mbeta1 .* G_db);
        C1 = C - step ./ (sqrt(Mm_c1) + decay) .* (options.beta1 * M_c1 + (1 - options.beta1) ./ mbeta1 .* G_c);
        Va1 = Va - step ./ (sqrt(Mm_va1) + decay) .* (options.beta1 * M_va1 + (1 - options.beta1) ./ mbeta1 .* G_va);
        Vb1 = Vb - step ./ (sqrt(Mm_vb1) + decay) .* (options.beta1 * M_vb1 + (1 - options.beta1) ./ mbeta1 .* G_vb);
    end

    %% Compute cost function
    obj = calculerErreur(XDep, Da1 * Va1, Db1 * Vb1, C1, Va1, Vb1, options.cas, options.alpha);
    relerr1 = abs(obj - obj0) / obj;
    hist(itTot) = relerr1;

    if obj / SSX < 1000 * eps
        relerr1 = 0;
        break;
    end

    %% Adjust the step size
    val_ref = 0.05; % This forces step increase when f_new < f_old
    if obj > obj0
        step = step / 2;
    elseif relerr1 < val_ref
        step = 1.2 * step;
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
C = renormaliser_dictionnaire2(C, norma);
% Recover the correct rank
[A, B, C] = best_rank(Da, Db, C, Va, Vb);
% Sort columns
[A, B, C] = triModes(A, B, C, 2);

end


%% Normalisation
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
%% Gradient clipping default =none
function [newgradient] = clipping(gradient,theta)

newgradient = gradient*theta/norm(gradient,2);
end
%% Choix du bon rang
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

function [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)

% Fonction triModes
%       /global
%
% Appels :
%       [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)
%       [Modestries] = triModes(Modes,num)
%
% Tri des colonnes des modes A,b, et C (ou du tableau de modes
% en fonction de la position croissante du maximum du mode n° num.
%
% En entrée :
%       A,B,C : les matrices initiales (le tableau Factors issu de parafac)
%       num   : 1, 2, ou 3
%
% En sortie :
%       Atrie,Btrie,Ctrie : les modes triés suivant la position du maximum
%

%   R. REDON
%   Laboratoire MIO, Université de Toulon
%   Créé le    : 06/12/2018
%   Modifié le : 06/12/2018


% Appel 1 :
% [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)
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
end % if (nargin==4)


% Appel 2 :
% [Modestries] = triModes(Modes,num)
if ( (nargin==2) && (nargout==1) && iscell(A) )
    num = B;
    [~,posmax] = max(A{num});
    [~,indtri] = sort(posmax);
    
    nA = length(A);
    Atrie = cell(1,nA);
    for k= 1:nA
        Atrie{k} = A{k}(:,indtri);
    end
end % if ( (nargin==2) && (nargout==1) && iscell(A) 


end

function C = renormaliser_dictionnaire2(Ctmp,norma)

C = Ctmp.*repmat(norma(:),[1 size(Ctmp,2)]);
end


