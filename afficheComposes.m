function displaycomponent(A, B, nbCompo, echelleRepresentation, axes)
% afficheComposes(A, B, nbCompo, echelleRepresentation, axes)
%
% A : Matrix of factors containing relative excitation spectra
% B : Matrix of factors containing relative emission spectra
% nbCompo : Number of components to display (should not exceed the number of columns in A and B)
% echelleRepresentation : Vector with 2 values defining the display scale (min and max bounds for color thresholds)
%     This parameter is in the form: [minBound maxBound]
%     Note that minBound is often chosen as 0
%     If this parameter is not provided, an automatic scale factor is applied to each figure
% axes : Vector with 4 values defining the x and y axes used for scaling
%     This parameter is in the form [xMin xMax yMin yMax]
%     If this parameter is not provided, default scaling is applied

Tm = 10;
MAP = zeros(Tm, 3);

% Define color map segments
coul1 = floor(Tm / 3);
coul2 = floor((Tm - coul1) / 2);
coul3 = floor(Tm - (coul1 + coul2));

% Define color gradient steps
pas1 = 1:-1/coul1:1/coul1;
pas2 = 1:-1/coul2:1/coul2;
pas3 = 1:-1/coul3:1/coul3;

MAP(1:coul1,:) = [ones(coul1,1), ones(coul1,1), pas1'];
MAP(coul1+1:coul1+coul2,:) = [ones(coul2,1), pas2', zeros(coul2,1)];
MAP(1+coul1+coul2:Tm,:) = [pas3', zeros(coul3,1), zeros(coul3,1)];

% Additional colormap settings
pas = 0.05;
intervalle = 1 : -pas : 0.1;
map2 = [intervalle' intervalle' intervalle'];

j = jet(128);
int = 0.8 : 0.2 / 15 : 1;
tmp = [int' int' ones(16, 1)];
j(1 : 16, :) = tmp;

sp = zeros(size(A(:, 1) * B(:, 1)'));

figure;

% Default axis scaling if not provided
if nargin < 5
    minX = 200; maxX = 550;
    minY = 250; maxY = 480;
    pasX = (maxX - minX) / size(A, 1);
    pasY = (maxY - minY) / size(B, 1);
    x = minX : pasX : maxX - pasX;
    y = minY : pasY : maxY - pasY;
else
    pasX = (axes(2) - axes(1)) / size(A, 1);
    pasY = (axes(4) - axes(3)) / size(B, 1);
    % [Additional scaling and plotting code]
end
