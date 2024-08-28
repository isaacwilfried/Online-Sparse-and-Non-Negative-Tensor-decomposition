function erreur = calculerErreur(XDep, A,B,C,Va,Vb, cas,alpha)
% ----------------------INPUT---------------------
%Xdep : Tenseur deplier suivant le mode 1
% A, B, C : Matrices facteurs
% Cas = 0 :Sans contrainte de parcimonie
% Cas = 1 : Avec contrainte de parcimonie

% ----------------------Output---------------------
% erreur : calcul de l'erreur quadratique

modele = A * transpose(pkr(C, B));
if(cas == 0)
    erreur=1/2*sum(sum((XDep - modele).^2));
end

if(cas == 1)
    l1_va = l1(Va);
    l1_vb = l1(Vb);
    erreur=1/2*sum(sum((XDep - modele).^2))+alpha * l1_va+ alpha *l1_vb ;

end