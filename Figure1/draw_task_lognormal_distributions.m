% Original MATLAB illustration-asset generator.
% This draws illustrative artwork, not a mission experiment.
% The complete final figure was composed and edited in PowerPoint;
% these assets alone do not reconstruct that final composition.
% plot.py renders the final authored illustration.

function outputPaths = draw_task_lognormal_distributions(outputDir, fontSize)
%DRAW_TASK_LOGNORMAL_DISTRIBUTIONS
% Draw workload distributions and actual workload realizations
% for three inspection tasks, and also draw one separate legend figure.
%
% Output:
%   1) task1_lognormal_distribution.png
%   2) task2_lognormal_distribution.png
%   3) task3_lognormal_distribution.png
%   4) actual_workload_legend.png

%% =========================================================
% Basic settings
% ==========================================================

scriptDir = fileparts(mfilename('fullpath'));

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(scriptDir, 'fig01_system_assets');
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
% Task workload distributions
% ==========================================================

taskIndex = 1:3;

% log(C_i) ~ N(mu_i, sigma_i^2)
mu = [-0.15, 0.42, 0.68];
sigma = [0.28, 0.46, 0.55];


%% Colors
colors = [ ...
      0, 114, 178; ...     % Task 1: blue
      0, 158, 115; ...     % Task 2: green
    139,  92, 246 ...      % Task 3: purple
    ] ./ 255;


%% Workload axis
workload = linspace(0.04, 5.0, 800);


%% =========================================================
% Actual workload realizations
% ==========================================================

realizedQuantile = [ ...
    0.12, ...   % Task 1
    0.60, ...   % Task 2
    0.85  ...   % Task 3
    ];

zRealized = sqrt(2) .* erfinv(2 .* realizedQuantile - 1);

realizedWorkload = exp(mu + sigma .* zRealized);


%% =========================================================
% Marker settings
% ==========================================================

hiddenMarkerColor = [0.18, 0.18, 0.18];
hiddenMarkerFaceColor = [1.00, 0.72, 0.08];
hiddenLineColor = [0.30, 0.30, 0.30];


%% Output paths
outputPaths = cell(1, 4);


%% =========================================================
% Draw the three distributions
% ==========================================================

for k = 1:numel(taskIndex)

    %% -----------------------------------------------------
    % Log-normal density
    % ------------------------------------------------------
    density = exp( ...
        -(log(workload) - mu(k)).^2 ./ ...
        (2 .* sigma(k).^2)) ./ ...
        (workload .* sigma(k) .* sqrt(2 .* pi));

    yMax = 1.18 .* max(density);


    %% -----------------------------------------------------
    % Actual workload realization
    % ------------------------------------------------------
    cActual = realizedWorkload(k);

    densityActual = exp( ...
        -(log(cActual) - mu(k)).^2 ./ ...
        (2 .* sigma(k).^2)) ./ ...
        (cActual .* sigma(k) .* sqrt(2 .* pi));


    %% =====================================================
    % Figure
    % =======================================================
    fig = figure( ...
        'Color', 'none', ...
        'Units', 'inches', ...
        'Position', [1, 1, 2.25, 1.58], ...
        'Visible', 'off');

    if isprop(fig, 'GraphicsSmoothing')
        fig.GraphicsSmoothing = 'on';
    end

    cleanupObject = onCleanup(@() close(fig)); %#ok<NASGU>


    %% =====================================================
    % Axes
    % =======================================================
    ax = axes( ...
        'Parent', fig, ...
        'Color', 'none', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 0.90 .* fontSize, ...
        'LineWidth', 0.9, ...
        'TickDir', 'out', ...
        'Layer', 'top', ...
        'Box', 'off');

    hold(ax, 'on');

    ax.XColor = [0.15, 0.15, 0.15];
    ax.YColor = [0.15, 0.15, 0.15];


    %% =====================================================
    % Known workload distribution
    % =======================================================
    fill(ax, ...
        [workload, fliplr(workload)], ...
        [density, zeros(size(density))], ...
        colors(k, :), ...
        'FaceAlpha', 0.16, ...
        'EdgeColor', 'none');

    plot(ax, ...
        workload, ...
        density, ...
        'Color', colors(k, :), ...
        'LineWidth', 2.2);


    %% =====================================================
    % Actual workload realization C_i
    % =======================================================
    plot(ax, ...
        [cActual, cActual], ...
        [0, densityActual], ...
        '--', ...
        'Color', hiddenLineColor, ...
        'LineWidth', 1.05);

    plot(ax, ...
        cActual, ...
        densityActual, ...
        'p', ...
        'MarkerSize', 11.5, ...
        'MarkerFaceColor', hiddenMarkerFaceColor, ...
        'MarkerEdgeColor', hiddenMarkerColor, ...
        'LineWidth', 1.15);


    %% =====================================================
    % Known distribution label
    % =======================================================
    text(ax, ...
        0.96, ...
        0.91, ...
        'Known distribution', ...
        'Units', 'normalized', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 0.80 .* fontSize, ...
        'FontWeight', 'normal', ...
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', ...
        'Color', colors(k, :));


    %% =====================================================
    % Hidden C_i label
    % =======================================================
    if k == 1

        labelX = cActual + 0.25;
        labelY = densityActual + 0.055 .* yMax - 0.2;
        horizontalAlignment = 'left';

    elseif k == 2

        labelX = cActual + 0.20;
        labelY = densityActual + 0.015 .* yMax - 0.1;
        horizontalAlignment = 'left';

    else

        labelX = cActual - 0.26;
        labelY = densityActual + 0.055 .* yMax - 0.09;
        horizontalAlignment = 'right';

    end

    text(ax, ...
        labelX, ...
        labelY, ...
        sprintf('Hidden $C_{%d}$', taskIndex(k)), ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 0.80 .* fontSize, ...
        'FontWeight', 'normal', ...
        'HorizontalAlignment', horizontalAlignment, ...
        'VerticalAlignment', 'bottom', ...
        'Color', hiddenMarkerColor);


    %% =====================================================
    % Axis settings
    % =======================================================
    xlim(ax, [0, 5]);
    ylim(ax, [0, yMax]);

    xticks(ax, [0, 5]);
    xticklabels(ax, {'0', '5'});

    yticks(ax, []);


    %% =====================================================
    % Title
    % =======================================================
    titleHandle = title(ax, ...
        sprintf('Task %d: $\\psi_{%d}(c)$', taskIndex(k), taskIndex(k)), ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 1.10 .* fontSize, ...
        'FontWeight', 'normal');

    titleHandle.Units = 'normalized';
    titleHandle.Position(2) = 0.96;


    %% =====================================================
    % X label
    % =======================================================
    xLabelHandle = xlabel(ax, ...
        'Workload $c$', ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', fontSize);

    xLabelHandle.Units = 'normalized';
    xLabelHandle.Position(1) = 0.50;
    xLabelHandle.Position(2) = -0.095;


    %% =====================================================
    % Y label
    % =======================================================
    ylabel(ax, ...
        'Probability', ...
        'Interpreter', 'tex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', fontSize);


    %% =====================================================
    % Axes position
    % =======================================================
    ax.Position = [ ...
        0.20, ...
        0.22, ...
        0.75, ...
        0.67 ...
        ];


    %% =====================================================
    % Export
    % =======================================================
    outputPaths{k} = fullfile( ...
        outputDir, ...
        sprintf('task%d_lognormal_distribution.png', taskIndex(k)));

    try
        exportgraphics(ax, ...
            outputPaths{k}, ...
            'Resolution', 320, ...
            'BackgroundColor', 'white');
    catch
        set(fig, ...
            'Color', 'white', ...
            'PaperUnits', 'inches', ...
            'PaperPosition', [0, 0, 2.25, 1.58], ...
            'PaperSize', [2.25, 1.58]);

        set(ax, 'Color', 'white');

        print(fig, outputPaths{k}, '-dpng', '-r320');
    end

    make_white_background_transparent(outputPaths{k});

    fprintf('Saved Task %d distribution to:\n%s\n', ...
        taskIndex(k), outputPaths{k});
    fprintf('Task %d: hidden workload C_%d = %.3f Gcycle\n\n', ...
        taskIndex(k), taskIndex(k), cActual);

    clear cleanupObject;

end


%% =========================================================
% Draw a separate figure for "Actual workload"
% ==========================================================

legendFig = figure( ...
    'Color', 'none', ...
    'Units', 'inches', ...
    'Position', [1, 1, 2.25, 0.65], ...
    'Visible', 'off');

if isprop(legendFig, 'GraphicsSmoothing')
    legendFig.GraphicsSmoothing = 'on';
end

cleanupLegend = onCleanup(@() close(legendFig)); %#ok<NASGU>

legendAx = axes( ...
    'Parent', legendFig, ...
    'Color', 'none', ...
    'Position', [0, 0, 1, 1], ...
    'Visible', 'off');

hold(legendAx, 'on');
xlim(legendAx, [0, 1]);
ylim(legendAx, [0, 1]);

plot(legendAx, ...
    0.18, 0.50, ...
    'p', ...
    'MarkerSize', 14, ...
    'MarkerFaceColor', hiddenMarkerFaceColor, ...
    'MarkerEdgeColor', hiddenMarkerColor, ...
    'LineWidth', 1.15);

text(legendAx, ...
    0.30, 0.50, ...
    'Actual workload', ...
    'FontName', 'Times New Roman', ...
    'FontSize', fontSize, ...
    'FontWeight', 'normal', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle', ...
    'Color', hiddenMarkerColor);

outputPaths{4} = fullfile(outputDir, 'actual_workload_legend.png');

try
    exportgraphics(legendAx, ...
        outputPaths{4}, ...
        'Resolution', 320, ...
        'BackgroundColor', 'white');
catch
    set(legendFig, ...
        'Color', 'white', ...
        'PaperUnits', 'inches', ...
        'PaperPosition', [0, 0, 2.25, 0.65], ...
        'PaperSize', [2.25, 0.65]);

    print(legendFig, outputPaths{4}, '-dpng', '-r320');
end

make_white_background_transparent(outputPaths{4});

fprintf('Saved separate legend figure to:\n%s\n\n', outputPaths{4});

clear cleanupLegend;

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