clc
clear all
close all
load('matrice_composants.mat')
M = csvreaderconcentration('Classeur1_concentration_theorique_02_11.csv');
X = M{:,:};

%% matrice pour t0
Aa_t0 = A(:,3);
Aa_t0(:,2) = A(:,4);
Aa_t0(:,3) = A(:,6);


Bb_t0 = B(:,3);
Bb_t0(:,2) = B(:,4);
Bb_t0(:,3) = B(:,6);

%% bloc t0
load('matrices_ntda_02_11_2020_2composants+BF_v.mat')
Lv = {Aa_t0,Bb_t0};
Ae = Da*Va;
Be = Db*Vb;
Ae(:,all(Ae == 0))=[] ;
Be(:,all(Be == 0))=[] ;

L = {Ae,Be};
[Le,ind]=permscalR(L,Lv,2);% correction de la permutation

% L = {Da*Va,Db*Vb};
% [Le,ind,Lvn]=permscal_under_over(L,Lv,1);% correction de la permutation

%% plot les 3 figures
for i=1:size(Le{1},2)
figure
subplot(2,1,1)
plot(Lv{1}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{1}(:,i),':','linewidth',2)
title('Bloc_t0');

subplot(2,1,2)
plot(Lv{2}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{2}(:,i),':','linewidth',2)
title('Bloc_t0');

end
%% affichage concentration

% concentration_t0 = X(:,3:4);
% 
% [concentration_t0,norma] = normalisation_matrice2(concentration_t0);
% concentration_t0=concentration_t0(1:20,:);
% [C0,norma] = normalisation_matrice2(C);
% figure
% subplot(2,1,1)
% plot(concentration_t0,'linewidth',2)
% subplot(2,1,2)
% plot(C0,':','linewidth',2)
% title('Concentration-bloct0');
% 
% 
% afficheComposes(Ae, Be, 4);
% afficheComposes(Aa_t0, Aa_t0, 4);





%% bloc t1_UA_DA
load('matrices_ntda_02_11_2020_3composants+BFonlineUavada.mat')
Aa_t1 = A(:,2:4);
Aa_t1(:,4) = A(:,6);

Bb_t1 = B(:,2:4);
Bb_t1(:,4) = B(:,6);

Lv = {Aa_t1,Bb_t1};
Ae = A_ntda;
Be = B_ntda;

Ae(:,all(Ae == 0))=[] ;
Be(:,all(Be == 0))=[] ;

% [Lv] = normalisation_tenseur2_cell(Lv);

L = {Ae,Be};
[Le,ind,Lvn]=permscal_under_over(L,Lv,1);% correction de la permutation
% [Le,ind]=permscalR(L,Lv,2);% correction de la permutation


for i=1:size(Le{1},2)
figure
subplot(2,1,1)
plot(Lv{1}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{1}(:,i),':','linewidth',2)
title('Online_Ua_Da_Va');

subplot(2,1,2)
plot(Lv{2}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{2}(:,i),':','linewidth',2)
title('Online_Ua_Da_Va');

end

concentration_t1 = X(:,2:4);
[concentration_t1,norma] = normalisation_matrice2(concentration_t1);
concentration_t1=concentration_t1(1:30,:);
[C0,norma] = normalisation_matrice2(C);
figure
subplot(2,1,1)
plot(concentration_t1,'linewidth',2)
subplot(2,1,2)
plot(C0,':','linewidth',2)
title('Concentration-bloct1-UaDaVa');

%% bloc t1_Suite
load('matrices_ntda_02_11_2020_3composants+BFonlineSuite.mat')
Aa_t1 = A(:,2:4);
Aa_t1(:,4) = A(:,6);

Bb_t1 = B(:,2:4);
Bb_t1(:,4) = B(:,6);


Lv = {Aa_t1,Bb_t1};
Ae = Ua*Va1;
Be = Ub*Vb1;

Ae(:,all(Ae == 0))=[] ;
Be(:,all(Be == 0))=[] ;

% [Lv] = normalisation_tenseur2_cell(Lv);

L = {Ae,Be};
[Le,ind,Lvn]=permscal_under_over(L,Lv,1);% correction de la permutation
% [Le,ind]=permscalR(L,Lv,2);% correction de la permutation

for i=1:size(Le{1},2)
figure
subplot(2,1,1)
plot(Lv{1}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{1}(:,i),':','linewidth',2)
title('Online_Suite');
subplot(2,1,2)
plot(Lv{2}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{2}(:,i),':','linewidth',2)
title('Online_Suite');

end

concentration_t1 = X(:,2:4);
[concentration_t1,norma] = normalisation_matrice2(concentration_t1);
concentration_t1=concentration_t1(1:30,:);
[C0,norma] = normalisation_matrice2(C);
figure
subplot(2,1,1)
plot(concentration_t1,'linewidth',2)
subplot(2,1,2)
plot(C0,':','linewidth',2)
title('Concentration-bloct1-Suite');

%% Concentration




%% bloc t3_UA_DA
load('matrices_ntda_02_11_2020_4composants+BFonlineUavada.mat')
Aa_t1 = A(:,1:4);
Aa_t1(:,5) = A(:,6);

Bb_t1 = B(:,1:4);
Bb_t1(:,5) = B(:,6);

Lv = {Aa_t1,Bb_t1};
Ae = Ua*Da*Va1;
Be = Ub*Db*Vb1;

Ae(:,all(Ae == 0))=[] ;
Be(:,all(Be == 0))=[] ;

% [Lv] = normalisation_tenseur2_cell(Lv);

L = {Ae,Be};
[Le,ind,Lvn]=permscal_under_over(L,Lv,1);% correction de la permutation
% [Le,ind]=permscalR(L,Lv,2);% correction de la permutation


for i=1:size(Le{1},2)
figure
subplot(2,1,1)
plot(Lv{1}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{1}(:,i),':','linewidth',2)
title('Online_Ua_Da_Va');

subplot(2,1,2)
plot(Lv{2}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{2}(:,i),':','linewidth',2)
title('Online_Ua_Da_Va');

end

concentration_t1 = X(:,1:4);
[concentration_t1,norma] = normalisation_matrice2(concentration_t1);
concentration_t1=concentration_t1(30:end,:);
[C0,norma] = normalisation_matrice2(C);
figure
subplot(2,1,1)
plot(concentration_t1,'linewidth',2)
subplot(2,1,2)
plot(C0,':','linewidth',2)
title('Concentration-bloct1-UaDaVa');

% bloc t3_Suite
load('matrices_ntda_02_11_2020_4composants+BFonlineSuite.mat')
Aa_t1 = A(:,1:4);
Aa_t1(:,5) = A(:,6);

Bb_t1 = B(:,1:4);
Bb_t1(:,5) = B(:,6);

Lv = {Aa_t1,Bb_t1};
Ae = Ua*Va1;
Be = Ub*Vb1;

Ae(:,all(Ae == 0))=[] ;
Be(:,all(Be == 0))=[] ;

% [Lv] = normalisation_tenseur2_cell(Lv);

L = {Ae,Be};
[Le,ind,Lvn]=permscal_under_over(L,Lv,1);% correction de la permutation
% [Le,ind]=permscalR(L,Lv,2);% correction de la permutation

for i=1:size(Le{1},2)
figure
subplot(2,1,1)
plot(Lv{1}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{1}(:,i),':','linewidth',2)
title('Online_Suite');
subplot(2,1,2)
plot(Lv{2}(:,i),'linewidth',2,'color','k')
hold on
plot(Le{2}(:,i),':','linewidth',2)
title('Online_Suite');

end

concentration_t1 = X(:,1:4);
[concentration_t1,norma] = normalisation_matrice2(concentration_t1);
concentration_t1=concentration_t1(30:end,:);
[C0,norma] = normalisation_matrice2(C);
figure
subplot(2,1,1)
plot(concentration_t1,'linewidth',2)
subplot(2,1,2)
plot(C0,':','linewidth',2)
title('Concentration-bloct1-Suite');

%% Concentration


