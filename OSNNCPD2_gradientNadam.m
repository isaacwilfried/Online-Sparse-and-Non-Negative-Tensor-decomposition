function [A,B,C,Ua,Ub,Da,Db,Va,Vb,hist,MU] = OSNNCPD2_gradientNadam(T,R,Dat0,Dbt0,Vat0,Vbt0,options)
%% OSNNCPD2 = Online and Sparse NonNegative CPD 2
% Fonction de CPD d'ordre 3 en ligne utilisant la methode d'apprentissage par
% dictionnaire et avec comme optimizer NADAM (variant de la descente de
% gradient stochastique ). Cette fonction permet de surestimer le rang au
% cas ou le rang est inconnu grace à la parcimonie imposée sur la fonction
% de cout. Une contrainte de nonnégativité est appliquée pour des données
% positives. Dans ce cas A=Ua*Da*Va est le dictionnaire obtenu à
% l'instant précedent. 
%Fonction de cout :
% 0.5*|| T - DA*VA \circ DB*VB \circ C||_F^2 + alpha||Va||_{1,1}+ alpha||Vb||_{1,1}
%
% 
% ----------------------INPUT---------------------
%Tensor T (3 order)
%Rank R 
%Dat0,Dbt0 : Dictionnaires obtenus à l'instant précedent
%Vat0,Vbt0 : Atomes obtenus à l'instant précedent

% ----------------OPTIONAL INPUT---------------------
%options: optional parameters
            % cas 0 : pas de parcimonie. Dans ce cas R est connu
            % Cas 1 : L1 norme sur VA,VB. Dans ce cas R est inconnu et il est
            % surestimé( par defaut)
            %alpha: Coefficient de pénalité
            % nonnegativite 0 : pas de contrainte de nonnégativité
            % nonnegativite 1 : avec contrainte de nonnégativité
            %beta1 est choisi entre [0.1;0.9], par defaut beta1 = 0.9
            %beta2 est choisi entre [0.1;0.9], par defaut beta2 = 0.9
             
% ----------------------Output---------------------
% DA,DB : Dictionnaires 
% VA,VB : Pois ou atome
% A : Matrice facteur (A=Ua*Da*Va)
% B : Matrice facteur  (B=Ub*Db*Vb)
% C : Matrice facteur
%hist : Erreur de reconstruction rélative pour chaque itération
%MU : pas d'apprentisssage pour chaque itération

% ----------------------Exemples---------------------
%   [A,B,C,Da,Db,Va,Vb,hist,MU] = OSNNCPD2_gradientNadam(T,R,Dat0,Dbt0,Vat0,Vbt0);
%   [A,B,C,Da,Db,Va,Vb,hist,MU] = OSNNCPD2_gradientNadam(T,R,Dat0,Dbt0,Vat0,Vbt0,[]);
%   [A,B,C,Da,Db,Va,Vb,hist,MU] = OSNNCPD2_gradientNadam(T,R,Dat0,Dbt0,Vat0,Vbt0,creerOptions(1, 1, 1.5, 0.6,0.7));

% ----------------------Fonctions utilisées---------------------
%C = renormaliser_dictionnaire2(C,norma) Renormaliser
%[X,norma] = normalisation_tenseur2(T) Nomaliser tenseur
%[A,B,C] = best_rank(Da,Db,C,Va,Vb) Trouver le bon rang
%[A,B,C] = triModes(A,B,C,2) Trie des composant par ordre croissant

% ----------------------REFs---------------------
% [1]Isaac Wilfried Sanou, Roland Redon, Xavier Luciani, and Stéphane Mounier. Online nonnegative
% and sparse canonical polyadic decomposition of fluorescence tensors. Chemometrics
% and Intelligent Laboratory Systems, 225 :104550, 2022.
% [2]saac Wilfried Sanou, Roland Redon, Xavier Luciani, and Stéphane Mounier. Online nonne-
% gative and sparse canonical polyadic decomposition of fluorescence tensors. Chemometrics
% and Intelligent Laboratory Systems, 225 :104550, 2022.
% [3] Timothy Dozat. Incorporating NESTEROV momentum into ADAM. Proceedings of 4th
% International Conference on Learning Representations, Workshop Track, 2016.

% see OSNNCPD1, SNNCPD
% Auteur : ISAAC SANOU
% Date de création : 01/02/2020
% Date de modification : 11/07/2022 

%% DEfault parameters
if(nargin < 7 || isempty(options))
    options = creerOptions(1, 1, 3, 0.9,0.9);
end

itTot = 0;
maxit = 20000;
obj0=inf;
decay=1e-6;

% normaliser
[T,norma] = normalisation_tenseur2(T);
% depliement
T1 = deplier(T, [1 2 3]);
T2 = deplier(T, [2 3 1]);
T3 = deplier(T, [3 1 2]);
%% initialisation

% ajout bruit à Dat0 et Dbt0
da = max(0,randn(size(T,1),R));
db = max(0,randn(size(T,2),R));

Da = Dat0*Vat0;
Db = Dbt0*Vbt0;

for i = 1:R
    if sum(Da(:,i))==0
        Da(:,i)=da(:,i);
        Db(:,i)=db(:,i);
    end
end


XDep = deplier(T, [1 2 3]);
SSX = sum(sum(XDep.^2));
% dictionnaire et poids
Ua= eye(size(T,1),size(T,1));
Va = eye(R,R);
Ub= eye(size(T,2),size(T,2));
Vb = eye(R,R);

%C  = randn(size(T,3),R);
 C  =max(0, zeros(size(T,3),R));


M_va=zeros(R,R);
M_vb =zeros(R,R);
M_c =zeros(size(T,3),R);
M_Ua =zeros(size(T,1),size(T,1));
M_Ub = zeros(size(T,2),size(T,2));

Mm_va=zeros(R,R);
Mm_vb =zeros(R,R);
Mm_c =zeros(size(T,3),R);
Mm_Ua =zeros(size(T,1),size(T,1));
Mm_Ub = zeros(size(T,2),size(T,2));

%derviée norme l_{1,1}
l1_va = ones(size(Va));
l1_vb = ones(size(Vb));

%% Choix de la contrainte ou non de parcimonie
if options.cas == 0
    step = 1e-1; %learning rate
end

if options.cas == 1
    step = 1e-3; %learning rate
end

relerr1=1;
crit = 10^-16;


while (relerr1> crit && itTot < maxit)
    %initialisation
    itTot = itTot + 1;
    
    %% Calcul du gradient
    [G_Ua,G_Ub,G_c,G_va,G_vb] = compute_gradientOnline(T1,T2,T3,Da,Ua,Ub,Db,Va,Vb,C,l1_va,l1_vb,options.cas,options.alpha);
    
    %% Mise à jour parametres NADAM
    M_va = options.beta1*M_va +(1-options.beta1)* G_va;
    M_vb = options.beta1*M_vb + (1-options.beta1)*G_vb;
    M_c = options.beta1*M_c + (1-options.beta1)*G_c;
    M_Ua = options.beta1*M_Ua + (1-options.beta1)*G_Ua;
    M_Ub = options.beta1*M_Ub + (1-options.beta1)*G_Ub;
    
    Mm_va = options.beta2*Mm_va +(1-options.beta2)* G_va.^2;
    Mm_vb = options.beta2*Mm_vb + (1-options.beta2)*G_vb.^2;
    Mm_c = options.beta2*Mm_c + (1-options.beta2)*G_c.^2;
    Mm_Ua = options.beta2*Mm_Ua + (1-options.beta2)*G_Ua.^2;
    Mm_Ub = options.beta2*Mm_Ub + (1-options.beta2)*G_Ub.^2;
    
    beta1_t = options.beta1.^itTot;
    mbeta1 = 1-beta1_t;
    M_va1 = M_va./mbeta1;
    M_vb1 = M_vb ./mbeta1;
    M_c1 = M_c ./ mbeta1;
    M_Ua1 = M_Ua ./mbeta1;
    M_Ub1 = M_Ub ./ mbeta1;
    
    beta2_t = options.beta2.^itTot;
    mbeta2 = 1-beta2_t;
    Mm_va1 = Mm_va./mbeta2;
    Mm_vb1 = Mm_vb ./mbeta2;
    Mm_c1 = Mm_c ./ mbeta2;
    Mm_Ua1 = Mm_Ua ./mbeta2;
    Mm_Ub1 = Mm_Ub ./ mbeta2;
    
    %% Mise à jour des matrices facteurs
    if  options.nonnegativite==1 %avec contrainte de nonnegativite
    
    Ua1 = max(Ua - step./(sqrt(Mm_Ua1)+decay).*(options.beta1*M_Ua1 + (1-options.beta1/1-beta1_t).*G_Ua),0); % pour assurer la nonnegativit�
    Va1 = max(Va - step./(sqrt(Mm_va1)+decay).*(options.beta1*M_va1 + (1-options.beta1/1-beta1_t).*G_va),0);
    
    Ub1 = max(Ub - step./(sqrt(Mm_Ub1)+decay).*(options.beta1*M_Ub1 + (1-options.beta1/1-beta1_t).*G_Ub),0);
    Vb1 = max(Vb - step./(sqrt(Mm_vb1)+decay).*(options.beta1*M_vb1 + (1-options.beta1/1-beta1_t).*G_vb),0);%%on peut supprimer le squart
    
    C1 = max(C - step./(sqrt(Mm_c1)+decay).*(options.beta1*M_c1 + (1-options.beta1/1-beta1_t).*G_c),0);
    
    end
    
    if  options.nonnegativite==0 % sans contrainte de nonnegativite
    Ua1 = (Ua - step./(sqrt(Mm_Ua1)+decay).*(options.beta1*M_Ua1 + (1-options.beta1/1-beta1_t).*G_Ua)); 
    Va1 = (Va - step./(sqrt(Mm_va1)+decay).*(options.beta1*M_va1 + (1-options.beta1/1-beta1_t).*G_va));

    Ub1 = (Ub - step./(sqrt(Mm_Ub1)+decay).*(options.beta1*M_Ub1 + (1-options.beta1/1-beta1_t).*G_Ub));
    Vb1 = (Vb - step./(sqrt(Mm_vb1)+decay).*(options.beta1*M_vb1 + (1-options.beta1/1-beta1_t).*G_vb));%%on peut supprimer le squart

    C1 = (C - step./(sqrt(Mm_c1)+decay).*(options.beta1*M_c1 + (1-options.beta1/1-beta1_t).*G_c));
    end
    
  %% calcul de la fonction de cout
  obj = calculerErreurOnline(T1, Da,Db,Ua1,Ub1,C1,Va1,Vb1, options.cas,options.alpha);

    relerr1 = abs(obj-obj0)/(obj);
    hist(itTot) = relerr1;
     if obj/SSX<1000*eps % Getting close to the machine uncertainty => stop
        relerr1 = 0;
        break;
    end

    %% adjust the step size
    val_ref=.05; %(cela force l'augmentation du pas lorsque f_new < f_old) %.005;
    if obj> obj0
        step=step/2;
        Ua = Ua1;
        Ub = Ub1;
        C = C1;
        Va = Va1;
        Vb = Vb1;
        obj0 = obj;
        
    elseif  relerr1 < val_ref
        step=1.2*step;
        Ua = Ua1;
        Ub = Ub1;
        C = C1;
        Va = Va1;
        Vb = Vb1;
        obj0 = obj;
        
    else
        Ua = Ua1;
        Ub = Ub1;
        C = C1;
        Va = Va1;
        Vb = Vb1;
        obj0 = obj;
        
        
    end

MU(itTot) = step;
end

%renormalisation
C = renormaliser_dictionnaire2(C,norma);
%recuperer le bon rang
[A,B,C] = best_rank(Ua*Da,Ub*Db,C,Va,Vb);
%tri des colonnes
[A,B,C] = triModes(A,B,C,2);

end





%% Normalisation
function [X,norma] = normalisation_tenseur2(T)

taille = size(T,3);
X = T;
norma = zeros(1,taille);

for k = 1:taille
    ff = T(:,:,k);
    norma(k) = max(abs(ff(:)));
    X(:,:,k) = T(:,:,k)./norma(k);
end
end
%% Gradient clipping default =none
function [newgradient] = clipping(gradient,theta)

newgradient = gradient*theta/norm(gradient,2);
end
%% Choix du bon rang
function [A,B,C] = best_rank(Da,Db,Ce,Va,Vb)
[row,col] = find(sum(Va)==0);
tf = isempty(col);
if tf == 1
    A = Da*Va;
    B = Db*Vb;
    C = Ce;
elseif tf == 0
    A = Da*Va;
    B =Db*Vb;
    C = Ce;
    A(:,col) = 0;
    B(:,col) = 0;
    C(:,col) = 0;
end
end


function [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)

% Fonction triModes
%       /global
%
% Appels :
%       [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)
%       [Modestries] = triModes(Modes,num)
%
% Tri des colonnes des modes A,b, et C (ou du tableau de modes
% en fonction de la position croissante du maximum du mode n° num.
%
% En entrée :
%       A,B,C : les matrices initiales (le tableau Factors issu de parafac)
%       num   : 1, 2, ou 3
%
% En sortie :
%       Atrie,Btrie,Ctrie : les modes triés suivant la position du maximum
%

%   R. REDON
%   Laboratoire MIO, Université de Toulon
%   Créé le    : 06/12/2018
%   Modifié le : 06/12/2018


% Appel 1 :
% [Atrie,Btrie,Ctrie] = triModes(A,B,C,num)
if ( (nargin==4) && (nargout==3) )
    switch num
        case 1
            [~,posmax] = max(A);
        case 2
            [~,posmax] = max(B);
        case 3
            [~,posmax] = max(C);
    end
    [~,indtri] = sort(posmax);
    Atrie = A(:,indtri);
    Btrie = B(:,indtri);
    Ctrie = C(:,indtri);
end % if (nargin==4)


% Appel 2 :
% [Modestries] = triModes(Modes,num)
if ( (nargin==2) && (nargout==1) && iscell(A) )
    num = B;
    [~,posmax] = max(A{num});
    [~,indtri] = sort(posmax);
    
    nA = length(A);
    Atrie = cell(1,nA);
    for k= 1:nA
        Atrie{k} = A{k}(:,indtri);
    end
end % if ( (nargin==2) && (nargout==1) && iscell(A) )


end

function C = renormaliser_dictionnaire2(Ctmp,norma)

C = Ctmp.*repmat(norma(:),[1 size(Ctmp,2)]);
end
