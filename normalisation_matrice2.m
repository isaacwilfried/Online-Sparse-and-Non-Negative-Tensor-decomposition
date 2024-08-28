function [X,norma] = normalisation_matrice2(T)

taille = size(T,2);
X = T;
norma = zeros(1,taille);

for k = 1:taille
    ff = T(:,k);
   norma(k) = max(abs(ff(:)));
    X(:,k) = T(:,k)./norma(k);
end

