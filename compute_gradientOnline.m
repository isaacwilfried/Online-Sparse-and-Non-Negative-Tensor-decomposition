function [G_Ua, G_Ub, G_c, G_va, G_vb] = compute_gradientOnline(T1, T2, T3, Da, Ua, Ub, Db, Va, Vb, C, l1_va, l1_vb, penalite, alpha)
%% compute_gradientOnline
% This function computes the gradient of the cost function for the online sparse 
% non-negative CPD algorithm with respect to the factor matrices Ua, Ub, C, Va, and Vb.
%
% ----------------------INPUT---------------------
% T1, T2, T3  : Tensor unfolded along modes 1, 2, 3 respectively.
% Ua, Ub, Da, Db : Dictionaries and transformation matrices.
% C          : Factor matrix corresponding to the third mode.
% Va, Vb     : Weights or atoms.
% penalite   : Penalty type
%              0 -> No sparsity (R is known)
%              1 -> L1 norm on Va, Vb (R is unknown and overestimated)
% alpha      : Penalty coefficient
% l1_va, l1_vb : Derivatives of the L1 norm for Va and Vb (matrices filled with ones)
%
% ----------------------OUTPUT---------------------
% G_Ua, G_Ub, G_c, G_va, G_vb : Gradients with respect to Ua, Ub, C, Va, and Vb respectively.

%% Compute intermediate Kronecker products for gradient calculation
L1 = pkr(C, Ub * Db); % Intermediate result for mode 1
L2 = pkr(Ua * Da, C); % Intermediate result for mode 2
L3 = pkr(Ub * Db, Ua * Da); % Intermediate result for mode 3

%% Gradient calculation based on penalty type
if penalite == 0 % Without sparsity
    % Gradient with respect to Ua
    G_Ua = -T1 * L1 * Va' + Ua * Da * Va * (L1)' * (Da * Va * L1')' + alpha * Ua;
    
    % Gradient with respect to Va
    G_va = -transpose(Ua * Da) * (-Ua * Da * Va * transpose(L1) + T1) * L1;

    % Gradient with respect to Ub
    G_Ub = -T2 * L2 * Vb' + Ub * Db * Vb * (L2)' * (Db * Vb * L2')' + alpha * Ub;
    
    % Gradient with respect to Vb
    G_vb = -transpose(Ub * Db) * (-Ub * Db * Vb * transpose(L2) + T2) * L2;
    
    % Gradient with respect to C
    G_c = -T3 * L3 + C * (L3)' * L3;
    
elseif penalite == 1 % With sparsity (L1 norm on Va and Vb)
    % Gradient with respect to Ua
    G_Ua = -(-Ua * Da * Va * L1' + T1) * transpose(Da * Va * L1');

    % Gradient with respect to Va
    G_va = -transpose(Ua * Da) * (-Ua * Da * Va * transpose(L1) + T1) * L1;
    G_va = G_va + alpha * l1_va; % Add L1 norm penalty for Va

    % Gradient with respect to Ub
    G_Ub = -(-Ub * Db * Vb * L2' + T2) * transpose(Db * Vb * L2');

    % Gradient with respect to Vb
    G_vb = -transpose(Ub * Db) * (-Ub * Db * Vb * transpose(L2) + T2) * L2;
    G_vb = G_vb + alpha * l1_vb; % Add L1 norm penalty for Vb
    
    % Gradient with respect to C
    G_c = -T3 * L3 + C * (L3)' * L3;
end

end
