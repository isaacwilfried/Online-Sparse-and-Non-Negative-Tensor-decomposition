function XDep = unfold(X, ordre)
% deplier Unfolds a 3D tensor into a matrix based on the specified order.
%
% XDep = deplier(X, ordre)
%
% Transforms the 3D tensor X into a matrix according to the specified order vector.
% The order vector [i j k] determines the unfolding direction, where 
% 1 <= i, j, k <= 3, and i != j != k.
%
% The resulting matrix has dimensions size(X, i) x (size(X, j) * size(X, k)).
%
% INPUTS:
%   X     : A 3D tensor to be unfolded.
%   ordre : A vector specifying the order of dimensions for unfolding, e.g., [1 2 3].
%
% OUTPUT:
%   XDep  : The unfolded matrix of size size(X, i) x (size(X, j) * size(X, k)).

% Permute the tensor X according to the specified order vector
X2 = permute(X, ordre);

% Reshape the permuted tensor into a 2D matrix
XDep = reshape(X2, size(X2, 1), size(X2, 2) * size(X2, 3));

end

