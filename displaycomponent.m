function displaycomponent(A, B, nbCompo, echelleRepresentation, axes)
% displaycomponent(A, B, nbCompo, echelleRepresentation, axes)
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

% Set default values if echelleRepresentation and axes are not provided
if nargin < 4 || isempty(echelleRepresentation)
    echelleRepresentation = []; % Empty, will be handled later
end

if nargin < 5 || isempty(axes)
    axes = []; % Empty, will be handled later
end

% Number of color segments for the colormap
Tm = 10;
MAP = zeros(Tm, 3); % Initialize the colormap matrix

% Define the segments for the color map
coul1 = floor(Tm / 3); % Number of steps in the first segment
coul2 = floor((Tm - coul1) / 2); % Number of steps in the second segment
coul3 = floor(Tm - (coul1 + coul2)); % Number of steps in the third segment

% Define the gradient steps for each color segment
pas1 = 1:-1/coul1:1/coul1;
pas2 = 1:-1/coul2:1/coul2;
pas3 = 1:-1/coul3:1/coul3;

% Assign the gradients to the colormap
MAP(1:coul1,:) = [ones(coul1,1), ones(coul1,1), pas1'];
MAP(coul1+1:coul1+coul2,:) = [ones(coul2,1), pas2', zeros(coul2,1)];
MAP(1+coul1+coul2:Tm,:) = [pas3', zeros(coul3,1), zeros(coul3,1)];

% Additional colormap settings
pas = 0.05; % Step size for the secondary colormap
intervalle = 1 : -pas : 0.1; % Range of values for the secondary colormap
map2 = [intervalle' intervalle' intervalle']; % Secondary colormap definition

% Adjust the 'jet' colormap
j = jet(128); % Start with the default 'jet' colormap
int = 0.8 : 0.2 / 15 : 1; % Define intensity levels
tmp = [int' int' ones(16, 1)]; % Create a gradient effect in the first 16 levels
j(1 : 16, :) = tmp; % Apply this gradient to the first 16 entries of the colormap

% Initialize a matrix for the spatial pattern
sp = zeros(size(A(:, 1) * B(:, 1)'));

% Create a new figure for plotting
figure;

% Default axis scaling if not provided
if isempty(axes)
    minX = 200; maxX = 550; % Default X-axis range
    minY = 250; maxY = 480; % Default Y-axis range
    pasX = (maxX - minX) / size(A, 1); % Calculate the step size for X-axis
    pasY = (maxY - minY) / size(B, 1); % Calculate the step size for Y-axis
    x = minX : pasX : maxX - pasX; % X-axis values
    y = minY : pasY : maxY - pasY; % Y-axis values
else
    pasX = (axes(2) - axes(1)) / size(A, 1); % Step size for X-axis based on provided axes
    pasY = (axes(4) - axes(3)) / size(B, 1); % Step size for Y-axis based on provided axes
    x = axes(1) : pasX : axes(2) - pasX; % X-axis values
    y = axes(3) : pasY : axes(4) - pasY; % Y-axis values
end

% Plot each component
for i = 1:nbCompo
    sp = A(:, i) * B(:, i)'; % Compute the spatial pattern for component i
    subplot(ceil(sqrt(nbCompo)), ceil(sqrt(nbCompo)), i); % Create a subplot for this component
    imagesc(x, y, sp); % Display the image
    if ~isempty(echelleRepresentation)
        caxis(echelleRepresentation); % Apply the specified color scale
    end
    colormap(j); % Apply the custom colormap
    title(['Component ' num2str(i)]); % Title each subplot
end

end
