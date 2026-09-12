% apply_curation.m  -  re-derive traces for the hand-curated footprint sets.
%
% Run apply_curation.py first; it writes curated_S.mat per plane.
%
% WHY THIS GOES BACK THROUGH EXTRACT
%   Two of the curation operations change what a trace means, and a plain
%   footprint-weighted average would get both wrong:
%     * pain plane B #11 was split into two somata. Their footprints touch,
%       so a weighted average of each would carry a large fraction of the
%       other's signal. EXTRACT's T-step solves all sources jointly and
%       demixes them.
%     * open-field plane B gained three footprints (N3, N4, N5) seeded as
%       gaussian disks. They need the S-step to take their real shape from
%       the movie, and N4/N5 sit within ~25 px of #4, so again they have to
%       be solved together rather than one at a time.
%
%   The motion-corrected movie is reused from <plane>_1.h5 rather than
%   recomputed (skip_steps_1_2), so this costs seconds of registration, not
%   minutes. Cell finding is off and every quality gate is disabled: the keep
%   or discard calls were made by eye, and re-applying the automatic filters
%   would silently undo them.
%
% ISOLATION - writes only into <plane folder>\curated\, never over
% final_analysis_results.mat and never into the other agent's output\.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
mini2p_toolbox_paths();

SESSIONS = { ...
  'openfield', 'D:\20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_15-27-52'; ...
  'pain',      'D:\20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_15-56-48' };
PLANES = {'A', 'B'};
% ONLY = {} runs everything; otherwise list 'session plane' keys, e.g.
% {'openfield B', 'pain B'}, to redo just those. SKIP_DONE leaves a plane
% alone if its curated result is already on disk.
ONLY = {};
SKIP_DONE = false;

o = struct();
o.ca_frame_rate_hz = 4.6054;           % measured from the Slice timestamps
o.extract_mode = 'curated';
o.trace_on_unbinned = true;
o.skip_steps_1_2 = true;               % reuse <plane>_1.h5
o.use_zscore_before_extract = false;
o.extract_preset = 'low_snr';
o.avg_cell_radius = 8;
o.skip_morph_filter = true;
o.n_frames_check = 0;
o.skip_video_01 = true;
o.skip_video_03 = true;
o.save_trace_files = true;

% No parpool here on purpose. The curated pass is one alternating update on
% an already motion-corrected movie, so workers buy nothing, and asking for
% a pool while the other agent's MATLAB holds one made this script exit
% silently with no output at all.

summary = {};
for s = 1:size(SESSIONS, 1)
    key = SESSIONS{s, 1};
    sdir = SESSIONS{s, 2};
    for k = 1:numel(PLANES)
        pl = PLANES{k};
        tagkey = [key, ' ', pl];
        if ~isempty(ONLY) && ~any(strcmpi(tagkey, ONLY)), continue; end
        pdir = fullfile(sdir, 'output_split', ['plane_', pl]);
        if SKIP_DONE && isfile(fullfile(pdir, 'curated', ...
                                        'final_analysis_results.mat'))
            fprintf('  %s : already done, skipped\n', tagkey);
            continue;
        end
        tif = fullfile(sdir, 'output_split', ['plane_', pl, '.tif']);
        sfile = fullfile(pdir, 'curated', 'curated_S.mat');
        if ~isfile(sfile)
            warning('missing %s - run apply_curation.py first', sfile);
            continue;
        end
        % step 2 derives <curr_header>_1.h5 from the tif name and looks for
        % it in the output folder, so the curated results go in a subfolder
        % and the h5 is linked in rather than copied (2.7 GB each)
        cout = fullfile(pdir, 'curated');
        h5_src = fullfile(pdir, ['plane_', pl, '_1.h5']);
        h5_dst = fullfile(cout, ['plane_', pl, '_1.h5']);
        if ~isfile(h5_dst)
            if ~isfile(h5_src)
                warning('missing %s', h5_src); continue;
            end
            [ok, msg] = system(sprintf('mklink /H "%s" "%s"', h5_dst, h5_src));
            if ok ~= 0 || ~isfile(h5_dst)
                warning(['hard link failed (%s); copying 2.7 GB instead. ' ...
                         'Delete %s when done.'], strtrim(msg), h5_dst);
                copyfile(h5_src, h5_dst);
            end
        end

        oo = o;
        oo.curated_S_file = sfile;
        fprintf('\n===== curated: %s plane %s =====\n', key, pl);
        t0 = tic;
        try
            pain_2plane_step2_extract(tif, cout, oo);
            f = fullfile(cout, 'final_analysis_results.mat');
            n = NaN;
            if isfile(f)
                S = load(f, 'output');
                n = size(S.output.spatial_weights, 3);
            end
            summary{end+1} = sprintf('  %-10s plane %s : %3d cells, %.0f s', ...
                                     key, pl, n, toc(t0)); %#ok<SAGROW>
        catch ME
            fprintf(2, '  FAILED: %s\n', ME.message);
            for e = 1:numel(ME.stack)
                fprintf(2, '      %s line %d\n', ME.stack(e).name, ...
                        ME.stack(e).line);
            end
            summary{end+1} = sprintf('  %-10s plane %s : FAILED (%s)', ...
                                     key, pl, ME.message); %#ok<SAGROW>
        end
    end
end

fprintf('\n==================== summary ====================\n');
for i = 1:numel(summary), disp(summary{i}); end
