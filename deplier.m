function XDep = deplier(X, ordre)

% XDep = deplier(X, ordre)
%
% Transforme le tenseur X de dimension 3 en matrice, dans le sens donné par
% le vecteur ordre de valeurs [i j k], 
% avec 1 <= i, j, k <= 3, et i != j != k.
%
% La matrice obtenue est de taille size(X, i) x (size(X, j) * size(X, k))

X2 = permute(X, ordre);
XDep = reshape(X2, size(X2, 1), size(X2, 2) * size(X2, 3)); 
