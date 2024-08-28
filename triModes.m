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

