function [A,B,C] = best_rank(Da,Db,Ce,Va,Vb)
% best_rank function calculates the factor matrices A, B, and C 
% based on input dictionaries and weights, considering sparsity.
%
% INPUTS:
% Da : Dictionary matrix for factor A
% Db : Dictionary matrix for factor B
% Ce : Factor matrix C (unchanged)
% Va : Weight matrix for factor A
% Vb : Weight matrix for factor B
%
% OUTPUTS:
% A : Factor matrix A
% B : Factor matrix B
% C : Factor matrix C (possibly with zeroed columns)

% Check for columns in Va that sum to zero
[row, col] = find(sum(Va) == 0);

% If no columns are fully zero, simply compute A and B
if isempty(col)
    A = Da * Va;
    B = Db * Vb;
    C = Ce;
% If there are zero columns, set corresponding columns in A, B, and C to zero
else
    A = Da * Va;
    B = Db * Vb;
    C = Ce;
    A(:, col) = 0;
    B(:, col) = 0;
    C(:, col) = 0;
end

end
