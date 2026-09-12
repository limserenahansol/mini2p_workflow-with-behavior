function p = mini2p_toolbox_paths()
% Resolve EXTRACT / NoRMCorre on this machine, and guarantee that every
% EXTRACT function resolves to the EXTRACT tree.
%
% OneDrive moved the old Documents\MATLAB\MATLAB\... folders.
%
% WHY ActSort IS DELIBERATELY *NOT* ON THE PATH HERE
% --------------------------------------------------
% ActSort ships its own forks of 20 EXTRACT helper files, 19 of which
% differ from EXTRACT's. addpath prepends, so adding ActSort after EXTRACT
% silently replaces EXTRACT's internals with ActSort's:
%
%     get_circularity_metrics.m  25 -> 46 lines   (extra 3rd arg 'parallel')
%     get_trace_noise.m          49 ->  6 lines
%     kappa_of_epsilon.m         22 ->  3 lines
%     estimate_noise_std.m, filter_images.m, smooth_images.m,
%     get_trace_snr.m, temporal_corruption.m, maybe_gpu.m,
%     find_spurious_cells.m, downsample_time.m, get_active_frames.m,
%     normalize_to_one.m, eps_func.m, select_indices.m, ndSparse.m,
%     get_free_mem.m, plot_cells_overlay.m, brewermap.m
%
% That shadowing is what made the 2026-09-11 split run die with
% "Not enough input arguments" at get_circularity_metrics line 12
% (= 'if parallel', the argument EXTRACT never passes), after already
% finding 73 / 60 cells with ActSort's noise and SNR estimators instead of
% EXTRACT's. So it corrupts results *before* it crashes.
%
% ActSort is a post-hoc manual cell sorter. Load it only after extraction
% is finished, with mini2p_actsort_path(), preferably in a fresh session.
%
% NoRMCorre has zero filename collisions with EXTRACT, so its order is
% irrelevant.

p = struct();
p.EXTRACT   = first_existing({
    'C:\Users\hsollim\OneDrive\Documents\MATLAB\MATLAB\EXTRACT-public'
    'C:\Users\hsollim\Documents\MATLAB\MATLAB\EXTRACT-public'
    });
p.NoRMCorre = first_existing({
    'C:\Users\hsollim\OneDrive\Documents\MATLAB\MATLAB\NoRMCorre-master'
    'C:\Users\hsollim\OneDrive\Documents\MATLAB\MATLAB\NoRMCorre'
    'C:\Users\hsollim\Documents\MATLAB\MATLAB\NoRMCorre'
    'C:\Users\hsollim\Documents\MATLAB\MATLAB\NoRMCorre-master'
    });
p.ActSort   = first_existing({
    'C:\Users\hsollim\OneDrive\Documents\MATLAB\MATLAB\ActSort-public-main'
    'C:\Users\hsollim\Documents\MATLAB\MATLAB\ActSort-public-main'
    });

if isempty(p.EXTRACT)
    error('EXTRACT-public not found. Clone https://github.com/schnitzer-lab/EXTRACT-public');
end
if isempty(p.NoRMCorre)
    error('NoRMCorre not found. Clone https://github.com/flatironinstitute/NoRMCorre');
end

% Strip ActSort if a saved path or an earlier script put it there.
if ~isempty(p.ActSort)
    on_path = strsplit(path, pathsep);
    stale = on_path(startsWith(lower(on_path), lower(p.ActSort)));
    if ~isempty(stale)
        rmpath(stale{:});
        fprintf('  removed %d ActSort folder(s) from the path (shadows EXTRACT)\n', ...
                numel(stale));
    end
end

addpath(genpath(p.NoRMCorre));
addpath(genpath(p.EXTRACT));

if exist('NoRMCorreSetParms', 'file') ~= 2
    error('NoRMCorreSetParms.m still not on path after addpath: %s', p.NoRMCorre);
end
if exist('extractor', 'file') ~= 2
    error('extractor.m not on path after addpath: %s', p.EXTRACT);
end

assert_no_extract_shadow(p.EXTRACT);

disp(['  EXTRACT:   ', p.EXTRACT]);
disp(['  NoRMCorre: ', p.NoRMCorre]);
end


function assert_no_extract_shadow(extract_root)
% Every .m in the EXTRACT tree must be what MATLAB actually calls.
% A silent shadow here changes the numbers, not just the control flow, so
% this is an error and not a warning.
f = dir(fullfile(extract_root, '**', '*.m'));
bad = {};
for i = 1:numel(f)
    [~, fn] = fileparts(f(i).name);
    w = which(fn);
    if ~isempty(w) && ~startsWith(lower(w), lower(extract_root))
        bad{end+1} = sprintf('%s -> %s', fn, w);  %#ok<AGROW>
    end
end
if ~isempty(bad)
    error('EXTRACT function(s) shadowed by another toolbox:\n  %s', ...
          strjoin(bad, sprintf('\n  ')));
end
fprintf('  path check: %d EXTRACT functions, none shadowed\n', numel(f));
end


function f = first_existing(cands)
f = '';
for i = 1:numel(cands)
    if isfolder(cands{i})
        f = cands{i};
        return;
    end
end
end
