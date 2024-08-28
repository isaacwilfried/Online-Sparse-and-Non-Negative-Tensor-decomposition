function [Le,ind]=permscalR(L,Lv,meth);
% L: cell arry contenant les matrices facteurs estimées
% Lv: cell arry contenant les vraies matrices facteurs
%meth: choix de la méthode de correction de l'eereeur de permuattion

N=size(L{1},2); %nombre de facteurs
Q=length(L); %ordre du tenseur
S=[];
Se=[];

% On normalise les facteurs par leur norme 2 et on les concatènes
for q=1:Q
    A=L{q};
    Av=Lv{q};   
    for n=1:N
        s=norm(A(:,n));
        if s==0
            s=1;
        end
        An(:,n)=A(:,n)/s;
        s=norm(Av(:,n));
        if s==0
            s=1;
        end       
        Avn(:,n)=Av(:,n)/s;
    end
    Sen=[Se;An];
    size(Sen);
    Sn=[S;Avn];
    clear An Avn
end

% On corrige l'erreur de permutation
if meth==1 % Methode 1 : Plus sure mais plus couteuse (impossible si plus de 9 facteurs cf aide de perms)
    P=perms(1:N);
    for i=1:size(P,1)
        P(i,:);
        r(i)=norm(abs(Sn)-abs(Sen(:,P(i,:))));
    end
    [~,pos]=min(r);
    ind=P(pos,:);
else % Méthode 2 : Plus rapide et moins couteuse mais peut ne pas marcher si facteurs très corrélés 
    D = Sen'*Sn;
    [l,c] = size(D);
    D = ones(l,c) - abs(D).^2;
    [Crit, ind] = min(D);  
end

for q=1:Q
    A=L{q}(:,ind);
    Av=Lv{q};
    for n=1:N
        s=norm(A(:,n));
        sv=norm(Av(:,n));
        signe=sign(sum(A(:,n)))*sign(sum(Av(:,n)));       
        if s~=0
            A(:,n)=signe*A(:,n)*sv/s;
        end
    end
    Le{q}=A;
%     clear Ae
end







