function [A,B,C,Da,Db,Va,Vb,hist,MU,grad] = NTDA_gradientNadam(T,R,cas,alpha,beta1,beta2)
%% NTDA = Nonnegative Tensor dictionary approximation
% min 0.5*|| T - DA*VA \circ DB*VB \circ C||_F^2
% ----------------------INPUT---------------------

%Tensor T 
%Rank R 
% ----------------OPTIONAL INPUT---------------------
% cas 0 : pas de parcimonie. Dans ce cas R est connu
% Cas 1 : L1 norme sur VA,VB. Dans ce cas R est inconnu et il est
% surestimé( par defaut)
% nonnegativite=1
%alpha: Coefficient de pénalité
%beta1 =0.9
%beta2=0.9
% Output :
% Dictionary DA,DB,C
% Weight : VA,VB
% A = Da*Va
% B = Db*Vb

[T,norma] = normalisation_tenseur2(T);

%% DEfault parameters

T1 = deplier(T, [1 2 3]);

T2 = deplier(T, [2 3 1]);

T3 = deplier(T, [3 1 2]);



maxit = 20000;
itTot = 0;
crit = 10^-16;
decay = 1e-6;
%% initialisation & Normalisation

% dictionnaire et poids
Da = max(0,randn(size(T,1),R));
Db = max(0,randn(size(T,2),R));
C  = max(0,randn(size(T,3),R));

% Da = eye(size(T,1),R);
% Db = eye(size(T,2),R);
% C  = eye(size(T,3),R);

Va = eye(R,R);
Vb = eye(R,R);

l1_va = ones(size(Va));
l1_vb = ones(size(Vb));

M_va=zeros(R,R);
M_vb =zeros(R,R);
M_c =zeros(size(T,3),R);
M_da =zeros(size(T,1),R);
M_db = zeros(size(T,2),R);

Mm_va=zeros(R,R);
Mm_vb =zeros(R,R);
Mm_c =zeros(size(T,3),R);
Mm_da =zeros(size(T,1),R);
Mm_db = zeros(size(T,2),R);



% fonction objective
XDep = deplier(T, [1 2 3]);
SSX = sum(sum(XDep.^2));


obj0 = inf;
%% Choix cas
if cas == 0
    step = 1e-1; %learning rate
end

if cas == 1
    step = 1e-6; %learning rate
end

relerr1=1;

while (relerr1> crit && itTot < maxit)
    %initialisation
    itTot = itTot + 1;
    % Calcul du gradient
    [G_da,G_db,G_c,G_va,G_vb] =  compute_gradient(T1,T2,T3,Da,Db,Va,Vb,C,l1_va,l1_vb,cas,alpha);
    

    M_va = beta1*M_va +(1-beta1)* G_va;
    M_vb = beta1*M_vb + (1-beta1)*G_vb;
    M_c = beta1*M_c + (1-beta1)*G_c;
    M_da = beta1*M_da + (1-beta1)*G_da;
    M_db = beta1*M_db + (1-beta1)*G_db;
    
    Mm_va = beta2*Mm_va +(1-beta2)* G_va.^2;
    Mm_vb = beta2*Mm_vb + (1-beta2)*G_vb.^2;
    Mm_c = beta2*Mm_c + (1-beta2)*G_c.^2;
    Mm_da = beta2*Mm_da + (1-beta2)*G_da.^2;
    Mm_db = beta2*Mm_db + (1-beta2)*G_db.^2;
    
    
    
    %MAJ
    beta1_t = beta1.^itTot;
    mbeta1 = 1-beta1_t;
    M_va1 = M_va./mbeta1;
    M_vb1 = M_vb ./mbeta1;
    M_c1 = M_c ./ mbeta1;
    M_da1 = M_da ./mbeta1;
    M_db1 = M_db ./ mbeta1;
    
    beta2_t = beta2.^itTot;
    mbeta2 = 1-beta2_t;
    
    Mm_va1 = Mm_va./mbeta2;
    Mm_vb1 = Mm_vb ./mbeta2;
    Mm_c1 = Mm_c ./ mbeta2;
    Mm_da1 = Mm_da ./mbeta2;
    Mm_db1 = Mm_db ./ mbeta2;
    
    Da1 = max(Da - step./(sqrt(Mm_da1)+decay).*(beta1*M_da1 + (1-beta1/1-beta1_t).*G_da),0); % pour assurer la nonnegativité
    Db1 = max(Db - step./(sqrt(Mm_db1)+decay).*(beta1*M_db1 + (1-beta1/1-beta1_t).*G_db),0);
    C1 = max(C - step./(sqrt(Mm_c1)+decay).*(beta1*M_c1 + (1-beta1/1-beta1_t).*G_c),0);
    Va1 = max(Va - step./(sqrt(Mm_va1)+decay).*(beta1*M_va1 + (1-beta1/1-beta1_t).*G_va),0);
    Vb1 = max(Vb - step./(sqrt(Mm_vb1)+decay).*(beta1*M_vb1 + (1-beta1/1-beta1_t).*G_vb),0);
    
    
    % calcul de obj
    obj=calculerErreur(XDep, Da1*Va1,Db1*Vb1,C1,Va1,Vb1, cas,alpha);
    hist(itTot) = obj;
    relerr1 = abs(obj-obj0)/(obj);

       if obj/SSX<1000*eps % Getting close to the machine uncertainty => stop
            relerr1 = 0;
            break;
        end
    val_ref=.05; %(cela force l'augmentation du pas lorsque f_new < f_old) %.005;
    if obj> obj0
        step=step/2;
        Da = Da1;
        Db = Db1;
        C = C1;
        Va = Va1;
        Vb = Vb1;
        obj0 = obj;
    elseif  relerr1 < val_ref
        step=1.2*step;
        Da = Da1;
        Db = Db1;
        C = C1;
        Va = Va1;
        Vb = Vb1;
        obj0 = obj;
    else
        Da = Da1;
        Db = Db1;
        C = C1;
        Va = Va1;
        Vb = Vb1;
        obj0 = obj;
        
    end
    
    MU(itTot) = step;
    grad(itTot) = norm([Da1(:);Va1(:);Db1(:);Vb1(:);C1(:)]);
    
end
C = renormaliser_dictionnaire2(C,norma);
[A,B,C] = best_rank(Da,Db,C,Va,Vb);
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
end % if ( (nargin==2) && (nargout==1) && iscell(A) 


end

function C = renormaliser_dictionnaire2(Ctmp,norma)

C = Ctmp.*repmat(norma(:),[1 size(Ctmp,2)]);
end