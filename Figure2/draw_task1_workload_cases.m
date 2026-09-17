% Original MATLAB illustration-asset generator.
% This draws illustrative artwork, not a mission experiment.
% The complete final figure was composed and edited in PowerPoint;
% these assets alone do not reconstruct that final composition.
% plot.py renders the final authored illustration.
% Renamed from draw_task_lognormal_distributions.m to match its
% original function name, draw_task1_workload_cases.

function outputPath = draw_task1_workload_cases(outputDir, fontSize)
%DRAW_TASK1_WORKLOAD_CASES
% Draw the workload distribution of Task 1 with two workload cases:
%   1) Task with low workload
%   2) Task with heavy workload
%
% Output:
%   task1_workload_cases.png

%% =========================================================
% Basic settings
% ==========================================================

scriptDir = fileparts(mfilename('fullpath'));

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(scriptDir, 'fig_task_workload_cases');
end

if nargin < 2 || isempty(fontSize)
    fontSize = 11;
end

validateattributes(fontSize, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'fontSize', 2);

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


%% =========================================================
% Task 1 workload distribution
% ==========================================================

taskIndex = 1;

% log(C_1) ~ N(mu_1, sigma_1^2)
mu = -0.15;
sigma = 0.28;

% Distribution color
distributionColor = [0, 114, 178] ./ 255;

% Workload case colors
firebrick = [178, 34, 34] ./ 255;
darkgreen = [0, 100, 0] ./ 255;

% Dark edge around the pentagram
markerEdgeColor = [0.18, 0.18, 0.18];

% Workload axis
workload = linspace(0.04, 5.0, 800);


%% =========================================================
% Workload cases
% ==========================================================

% Illustrative quantiles
% Move the low workload point slightly to the left
lowQuantile = 0.18;
heavyQuantile = 0.85;

zLow = sqrt(2) .* erfinv(2 .* lowQuantile - 1);
zHeavy = sqrt(2) .* erfinv(2 .* heavyQuantile - 1);

cLow = exp(mu + sigma .* zLow);
cHeavy = exp(mu + sigma .* zHeavy);


%% =========================================================
% Lognormal density
% ==========================================================

density = exp( ...
    -(log(workload) - mu).^2 ./ ...
    (2 .* sigma.^2)) ./ ...
    (workload .* sigma .* sqrt(2 .* pi));

yMax = 1.20 .* max(density);


%% =========================================================
% Density values at the two workload cases
% ==========================================================

densityLow = exp( ...
    -(log(cLow) - mu).^2 ./ ...
    (2 .* sigma.^2)) ./ ...
    (cLow .* sigma .* sqrt(2 .* pi));

densityHeavy = exp( ...
    -(log(cHeavy) - mu).^2 ./ ...
    (2 .* sigma.^2)) ./ ...
    (cHeavy .* sigma .* sqrt(2 .* pi));


%% =========================================================
% Figure
% ==========================================================

fig = figure( ...
    'Color', 'none', ...
    'Units', 'inches', ...
    'Position', [1, 1, 3.15, 2.05], ...
    'Visible', 'off');

if isprop(fig, 'GraphicsSmoothing')
    fig.GraphicsSmoothing = 'on';
end

cleanupObject = onCleanup(@() close(fig)); %#ok<NASGU>


%% =========================================================
% Axes
% ==========================================================

ax = axes( ...
    'Parent', fig, ...
    'Color', 'none', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 1.35 .* fontSize, ...
    'LineWidth', 0.9, ...
    'TickDir', 'out', ...
    'Layer', 'top', ...
    'Box', 'off');

hold(ax, 'on');

ax.XColor = [0.15, 0.15, 0.15];
ax.YColor = [0.15, 0.15, 0.15];


%% =========================================================
% Workload distribution
% ==========================================================

fill(ax, ...
    [workload, fliplr(workload)], ...
    [density, zeros(size(density))], ...
    distributionColor, ...
    'FaceAlpha', 0.16, ...
    'EdgeColor', 'none');

plot(ax, ...
    workload, ...
    density, ...
    'Color', distributionColor, ...
    'LineWidth', 2.2);


%% =========================================================
% Low workload point
% ==========================================================

plot(ax, ...
    cLow, ...
    densityLow, ...
    'p', ...
    'MarkerSize', 10.5, ...
    'MarkerFaceColor', darkgreen, ...
    'MarkerEdgeColor', markerEdgeColor, ...
    'LineWidth', 1.10);

text(ax, ...
    cLow + 0.20, ...
    densityLow, ...
    sprintf('Task with low workload\nworkload = %.1f', cLow), ...
    'FontName', 'Times New Roman', ...
    'FontSize', 1.17 .* fontSize, ...
    'FontWeight', 'normal', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle', ...
    'Color', darkgreen);


%% =========================================================
% Heavy workload point
% ==========================================================

plot(ax, ...
    cHeavy, ...
    densityHeavy, ...
    'p', ...
    'MarkerSize', 10.5, ...
    'MarkerFaceColor', firebrick, ...
    'MarkerEdgeColor', markerEdgeColor, ...
    'LineWidth', 1.10);

text(ax, ...
    cHeavy + 0.20, ...
    densityHeavy, ...
    sprintf('Task with heavy workload\nworkload = %.1f', cHeavy), ...
    'FontName', 'Times New Roman', ...
    'FontSize', 1.17 .* fontSize, ...
    'FontWeight', 'normal', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle', ...
    'Color', firebrick);


%% =========================================================
% Axis settings
% ==========================================================

xlim(ax, [0, 5]);
ylim(ax, [0, yMax]);

xticks(ax, [0, 5]);
xticklabels(ax, {'0', '5'});

yticks([]);


%% =========================================================
% Title
% ==========================================================

titleHandle = title(ax, ...
    sprintf('Task %d: $psi_{%d}(c)$', taskIndex, taskIndex), ...
    'Interpreter', 'latex', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 1.65 .* fontSize, ...
    'FontWeight', 'normal');

titleHandle.Units = 'normalized';
titleHandle.Position(2) = 0.96;


%% =========================================================
% X label
% ==========================================================

xLabelHandle = xlabel(ax, ...
    'Workload $c$', ...
    'Interpreter', 'latex', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 1.50 .* fontSize);

xLabelHandle.Units = 'normalized';
xLabelHandle.Position(1) = 0.50;
xLabelHandle.Position(2) = -0.095;


%% =========================================================
% Y label
% ==========================================================

ylabel(ax, ...
    'Probability', ...
    'Interpreter', 'tex', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 1.50 .* fontSize);


%% =========================================================
% Axes position
% ==========================================================

ax.Position = [ ...
    0.16, ...
    0.21, ...
    0.80, ...
    0.68 ...
    ];


%% =========================================================
% Export
% ==========================================================

outputPath = fullfile( ...
    outputDir, ...
    'task1_workload_cases.png');

try
    exportgraphics(ax, ...
        outputPath, ...
        'Resolution', 320, ...
        'BackgroundColor', 'white');
catch
    set(fig, ...
        'Color', 'white', ...
        'PaperUnits', 'inches', ...
        'PaperPosition', [0, 0, 3.15, 2.05], ...
        'PaperSize', [3.15, 2.05]);

    set(ax, 'Color', 'white');

    print(fig, outputPath, '-dpng', '-r320');
end

make_white_background_transparent(outputPath);

fprintf('Saved Task 1 workload cases to:\n%s\n', outputPath);
fprintf('Low workload C_1 = %.1f\n', cLow);
fprintf('Heavy workload C_1 = %.1f\n', cHeavy);

clear cleanupObject;

end


function make_white_background_transparent(outputPath)
%MAKE_WHITE_BACKGROUND_TRANSPARENT
% Replace the white background with transparency while
% keeping antialiased lines and text.

[imageData, colorMap] = imread(outputPath);

if isempty(colorMap)

    if ndims(imageData) == 2
        imageData = repmat(imageData, 1, 1, 3);
    end

    if isinteger(imageData)
        rgb = double(imageData) ./ double(intmax(class(imageData)));
    else
        rgb = double(imageData);
    end

else

    rgb = ind2rgb(imageData, colorMap);

end

alphaData = max(1 - rgb, [], 3);
alphaData(alphaData < 1 / 255) = 0;

safeAlpha = max(alphaData, eps);

foreground = 1 - bsxfun(@rdivide, 1 - rgb, safeAlpha);
foreground = min(max(foreground, 0), 1);

transparent = alphaData == 0;

for channel = 1:3
    plane = foreground(:, :, channel);
    plane(transparent) = 1;
    foreground(:, :, channel) = plane;
end

pixelsPerMeter = round(320 / 0.0254);

imwrite( ...
    foreground, ...
    outputPath, ...
    'Alpha', alphaData, ...
    'XResolution', pixelsPerMeter, ...
    'YResolution', pixelsPerMeter, ...
    'ResolutionUnit', 'meter');

end