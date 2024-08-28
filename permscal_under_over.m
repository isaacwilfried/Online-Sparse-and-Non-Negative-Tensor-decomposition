function [Le,ind,Lvn]=permscal_under_over(L,Lv,meth);
% L = cell estime
% Lv = cell vrai

%%
N=length(L); %Ordre du tenseur
R=size(L{1},2);% rang des matrices estim�es
Rv=size(Lv{1},2);% vrai rang
X=[];
Xv=[];
Se=[];
S=[];

%% Normalisation et concatenation de L et Lv
for n=1:N
    A=L{n};
    Av=Lv{n};
    for i=1:R
%         s=sum(A(:,i));
        s=norm(A(:,i));
        An(:,i)=A(:,i)/s;
    end
    for k=1:Rv
%         s=sum(Av(:,k));
        s=norm(Av(:,k));
        Avn(:,k)=Av(:,k)/s;
    end
    X=[X;An];
    Xv=[Xv;Avn];
    Sen=[Se;An];
%     size(Sen);
    Sn=[S;Avn];
    Len{n} = An;
    Lvn{n} = Avn;
    clear An Avn
end
% [Crit,ind,Hvn,Hn] = mesure(H_vrai,H_est)

%% On corrige l'erreur de permutation
% %%methode 2
D = X'*Xv;
[l,c] = size(D);
D = ones(l,c) - abs(D).^2;
[Crit ind] = min(D);

% %methode 1 (il faut resize Sn)

% rang = R-Rv;
% z= repmat(0,size(Sn,1),Sn);
% size(z)
% .+
% Nf=size(L{1},2); %nombre de facteurs
% P=perms(1:Nf);
% for i=1:size(P,1)
%     P(i,:);
%     r(i)=norm(abs(Sn)-abs(Sen(:,P(i,:))));
% end
% [~,pos]=min(r);
% ind=P(pos,:);

Q = N;
ind;
for q=1:Q
    A=L{q}(:,ind);
    Av=Lv{q};
    size(A);
    for n=1:Rv
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

% Hvn=Xv;
% Hn=X(:,ind);
% Hn=Hn.*repmat(sign(Hvn(1,:)).*sign(Hn(1,:)),[size(Hn,1) 1]);
% Lvn{i}=Hvn;
% Len{i}=Hn;