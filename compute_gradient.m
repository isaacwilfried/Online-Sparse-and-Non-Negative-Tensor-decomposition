function [G_da,G_db,G_c,G_va,G_vb] = compute_gradient(T1,T2,T3,Da,Db,Va,Vb,C,l1_va,l1_vb,cas,alpha)
%% compute_gradient Computes the gradient of the cost function
%
% ----------------------INPUT---------------------
% T1, T2, T3 : Tensor unfolded along modes 1, 2, and 3, respectively.
% Da, Db : Dictionaries.
% C : Factor matrix.
% Va, Vb : Weights or atoms.
% cas : Determines the type of sparsity constraint:
%       0 : No sparsity. In this case, R is known.
%       1 : L1 norm on Va, Vb. In this case, R is unknown and overestimated (default).
% alpha : Penalty coefficient for L1 regularization.
% l1_va, l1_vb : Derivatives of the L1 norm of Va and Vb (matrices filled with ones).
%
% ----------------------OUTPUT---------------------
% G_da, G_db, G_c, G_va, G_vb : Gradients of Da, Db, C, Va, Vb respectively.
    
%% Initialization
% Compute Khatri-Rao products needed for the gradient computation
L1 = pkr(C, Db*Vb);
L2 = pkr(Da*Va, C);
L3 = pkr(Db*Vb, Da*Va);

%% Gradient computation based on sparsity constraint
if cas == 0  % Without sparsity
    % Compute gradients G_da and G_va
    G_da = -T1 * L1*Va' + Da*Va*(L1)'*L1*Va';
    G_va = - Da' * (-Da * Va * L1' + T1) * L1 ;
    
    % Compute gradients G_db and G_vb
    G_db = -T2 * L2*Vb' + Db*Vb*(L2)'*L2*Vb';
    G_vb = - Db' * (-Db * Vb * L2' + T2) * L2 ;  

    % Compute gradient G_c
     G_c = -T3*L3 + C*(L3)'*L3;    

elseif cas == 1  % With sparsity (L1 norm on Va and Vb)
    % Compute gradients G_da and G_va with L1 regularization on Va
    G_da = -T1 * L1*Va' + Da*Va*(L1)'*L1*Va';
    G_va = - Da' * (-Da * Va * L1' + T1) * L1 ;
    G_va = G_va + alpha * l1_va;  % Add L1 regularization term

    % Compute gradients G_db and G_vb with L1 regularization on Vb
    G_db = -T2 * L2*Vb' + Db*Vb*(L2)'*L2*Vb';
    G_vb = - Db' * (-Db * Vb * L2' + T2) * L2 ;  
    G_vb = G_vb + alpha * l1_vb;  % Add L1 regularization term

    % Compute gradient G_c
    G_c = -T3 * L3 + C * (L3)' * L3;
end

end
