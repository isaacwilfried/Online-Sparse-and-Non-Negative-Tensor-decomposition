function erreur = calculerErreurOnline(XDep, Da, Db, Ua, Ub, C, Va, Vb, cas, alpha)
%% calculerErreurOnline 
% This function computes the reconstruction error for the online CPD model.
% Depending on the value of 'cas', it can also include a sparsity penalty term.

% ----------------------INPUT---------------------
% XDep : Unfolded tensor along mode 1.
% Da, Db : Dictionaries.
% Ua, Ub : Transformation matrices.
% C     : Factor matrix corresponding to the third mode.
% Va, Vb : Weights or atoms.
% cas   : 0 -> Without sparsity constraint.
%         1 -> With sparsity constraint (L1 norm on Va, Vb).
% alpha : Penalty coefficient for the L1 norm.

% ----------------------OUTPUT---------------------
% erreur : Computed quadratic error (with or without L1 penalty).

%% Reconstruct the tensor from the factor matrices
A = Ua * Da * Va;
B = Ub * Db * Vb;
modele = A * transpose(pkr(C, B));

%% Calculate the error based on the selected mode (cas)
if cas == 0
    % Without sparsity constraint
    erreur = 1/2 * sum(sum((XDep - modele).^2)); % Quadratic error
elseif cas == 1
    % With sparsity constraint (L1 norm)
    l1_va = l1(Va);
    l1_vb = l1(Vb);
    erreur = 1/2 * sum(sum((XDep - modele).^2)) + alpha * l1_va + alpha * l1_vb;
end

end

%% Auxiliary function to calculate L1 norm of a matrix
function l1_norm = l1(V)
    l1_norm = sum(sum(abs(V)));
end
