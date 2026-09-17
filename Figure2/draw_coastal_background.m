% Original MATLAB illustration-asset generator.
% This draws illustrative artwork, not a mission experiment.
% The complete final figure was composed and edited in PowerPoint;
% these assets alone do not reconstruct that final composition.
% plot.py renders the final authored illustration.

function outputPath = draw_coastal_background(outputPath, backgroundAlpha)
%DRAW_COASTAL_BACKGROUND Draw a text-free land-and-coast background.
%   OUTPUTPATH = DRAW_COASTAL_BACKGROUND() writes
%   fig01_system_background.png next to this file at 320 dpi with an
%   overall opacity of 0.5.
%
%   OUTPUTPATH = DRAW_COASTAL_BACKGROUND(OUTPUTPATH) writes the PNG to the
%   specified path. The two filled regions are land and coastal water; the
%   shoreline and waves are drawn as line objects.
%
%   OUTPUTPATH = DRAW_COASTAL_BACKGROUND(OUTPUTPATH, BACKGROUNDALPHA) sets
%   the opacity of the complete PNG. BACKGROUNDALPHA must be in [0, 1].

if nargin < 1 || isempty(outputPath)
    scriptDir = fileparts(mfilename('fullpath'));
    outputPath = fullfile(scriptDir, 'fig01_system_background.png');
end

if nargin < 2 || isempty(backgroundAlpha)
    backgroundAlpha = 0.8;
end
validateattributes(backgroundAlpha, {'numeric'}, ...
    {'scalar', 'real', 'finite', '>=', 0, '<=', 1}, ...
    mfilename, 'backgroundAlpha', 2);

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

backgroundAlpha = 0.8;

% The displayed 10:7.2 range follows the requested coastal crop.
fig = figure( ...
    'Color', 'white', ...
    'Units', 'inches', ...
    'Position', [1, 1, 7, 3.6], ...
    'Visible', 'off');
if isprop(fig, 'GraphicsSmoothing')
    fig.GraphicsSmoothing = 'on';
end
cleanupObject = onCleanup(@() close(fig)); %#ok<NASGU>

ax = axes( ...
    'Parent', fig, ...
    'Position', [0, 0, 1, 1], ...
    'XLim', [0, 10], ...
    'YLim', [0, 7.2], ...
    'DataAspectRatio', [1, 1, 1], ...
    'Visible', 'off');
hold(ax, 'on');

coastColor = [221, 244, 250] / 255;
landColor = [234, 241, 198] / 255;
shoreColor = [74, 143, 139] / 255;
waveColor = [150, 217, 234] / 255;

% Coastal water fills the full canvas first.
rectangle(ax, ...
    'Position', [0, 0, 14, 7.2], ...
    'FaceColor', coastColor, ...
    'EdgeColor', 'none');

% A smooth sinusoidal shoreline separates the two regions.
shoreY = linspace(0, 7.2, 220);
shoreX = 4.78 + 0.18 .* sin(2 .* pi .* shoreY ./ 1.65 + 0.35);
landX = [0, shoreX, 0];
landY = [0, shoreY, 7.2];
patch(ax, landX, landY, landColor, 'EdgeColor', 'none');
plot(ax, shoreX, shoreY, ...
    'Color', shoreColor, ...
    'LineWidth', 1.7);

% Repeated light waves create the same cartoon coastal texture.
waveX = linspace(5.45, 13.55, 420);
waveLevels = [0.75, 1.78, 2.83, 3.88, 4.93];
for waveY = waveLevels
    waveCurve = waveY + 0.075 .* sin(2 .* pi .* (waveX - 5.45) ./ 0.82);
    plot(ax, waveX, waveCurve, ...
        'Color', waveColor, ...
        'LineWidth', 0.9);
end

% A thin frame gives a clean boundary when the PNG is placed in LaTeX.
%rectangle(ax, ...
%    'Position', [0.02, 0.02, 13.96, 7.16], ...
%    'FaceColor', 'none', ...
%    'EdgeColor', shoreColor, ...
%    'LineWidth', 0.0);%

axis(ax, 'off');
drawnow;

% exportgraphics is available in MATLAB R2020a and later. The print
% fallback keeps the function usable in older MATLAB releases.
try
    exportgraphics(ax, outputPath, ...
        'Resolution', 320, ...
        'BackgroundColor', 'white');
catch
    set(fig, ...
        'PaperUnits', 'inches', ...
        'PaperPosition', [0, 0, 7, 3.6], ...
        'PaperSize', [7, 3.6]);
    print(fig, outputPath, '-dpng', '-r320');
end

% Write a real alpha channel so the exported PNG remains translucent when
% it is placed over another figure, slide, or document background.
[imageData, colorMap] = imread(outputPath);
alphaData = backgroundAlpha .* ones( ...
    size(imageData, 1), size(imageData, 2));
pixelsPerMeter = round(320 / 0.0254);
if isempty(colorMap)
    imwrite(imageData, outputPath, ...
        'Alpha', alphaData, ...
        'XResolution', pixelsPerMeter, ...
        'YResolution', pixelsPerMeter, ...
        'ResolutionUnit', 'meter');
else
    imwrite(imageData, colorMap, outputPath, ...
        'Alpha', alphaData, ...
        'XResolution', pixelsPerMeter, ...
        'YResolution', pixelsPerMeter, ...
        'ResolutionUnit', 'meter');
end

fprintf('Saved coastal background (alpha %.2f) to: %s\n', ...
    backgroundAlpha, outputPath);
end
