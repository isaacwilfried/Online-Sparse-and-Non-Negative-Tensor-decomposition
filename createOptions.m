function options = createOptions(cas, nonnegativite, alpha, beta1, beta2, step)
% creerOptions Creates a structure of options for the SNNCPD algorithm.
%
% INPUTS:
%   cas           - Determines the type of sparsity constraint:
%                   0: No sparsity, the rank R is known.
%                   1: L1 norm on VA, VB, the rank R is unknown and overestimated (default).
%   nonnegativite - Non-negativity constraint:
%                   0: No non-negativity constraint.
%                   1: Apply non-negativity constraint.
%   alpha         - Penalty coefficient for the L1 regularization.
%   beta1         - First momentum parameter for the NADAM optimizer, typically between [0.1, 0.9], default is 0.9.
%   beta2         - Second momentum parameter for the NADAM optimizer, typically between [0.1, 0.9], default is 0.9.
%   step          - Learning rate for the optimization process.
%
% OUTPUT:
%   options - Structure containing the specified options for the SNNCPD algorithm.

% Create the options structure with the specified parameters
options = struct('cas', cas, 'nonnegativite', nonnegativite, ...
                 'alpha', alpha, 'beta1', beta1, 'beta2', beta2, ...
                 'step', step);

end
