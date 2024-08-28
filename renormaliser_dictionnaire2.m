function C = renormaliser_dictionnaire2(Ctmp,norma)

C = Ctmp.*repmat(norma(:),[1 size(Ctmp,2)]);
