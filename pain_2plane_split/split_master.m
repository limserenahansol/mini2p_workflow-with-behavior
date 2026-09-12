% split_master.m  -  plane-split EXTRACT, isolated from the other agent's run.
%
% Open and press F5.
%
% WHY THIS EXISTS SEPARATELY
%   The main pipeline (pain_2plane_pipeline\) max-projects the two ETL depths
%   into one movie and is being run by another agent. Measured on this data
%   (EXTRACT_QC_20260911.md):
%     * the depths hold different somata - 5 of 17 cells at one depth sit
%       within a cell radius of a cell at the other, so ~29 % get merged;
%     * max() is nonlinear and neither depth dominates, so the projected
%       value flips between the two neurons in 50 % of consecutive frames.
%   This version keeps the depths apart instead.
%
% ISOLATION - nothing here writes into the other agent's paths
%   code    C:\...\cursor\pain_2plane_split\        (copies, not shared)
%   output  <session>\output_split\                 (never <session>\output)
%   GPU off and parpool capped, so the other MATLAB keeps its workers.
%
% WHAT CHANGED VERSUS THE SHARED PIPELINE
%   1  depths split, not max-projected            -> one neuron per trace
%   2  use_zscore_before_extract = false           -> (F-F0)/F0 is real dF/F
%   3  trace_on_unbinned = true                    -> find cells on the binned
%      movie, then reuse the footprints at 4.605 Hz so the [-3,-1] s and
%      [1,3] s windows hold 9 samples instead of 2
%   4  preset low_snr, avg_cell_radius 8           -> matches the measured
%      radius of 7.9 px and the measured SNR of 0.75

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);
mini2p_toolbox_paths();

% ---------------- what to run ----------------
% Both sessions image the SAME two depths - etl1data.csv is byte-identical
% between them (ETL 65.0/139.6 um and 15.0/89.6 um). The "(100_50um)" in the
% open-field folder name is not the depths. So plane A of one session and
% plane A of the other are the same optical plane, and cells can in principle
% be matched across the two.
%
% Information-CHA.txt confirms the interleave rather than leaving it inferred:
%   Slice 2, Frames_per_slice 1  -> the two depths alternate frame by frame
%   openfield  Loops 1500, Save Frames 3000  -> 1500 frames/plane, 325.7 s
%   pain       Save Frames 6000              -> 3000 frames/plane, 651.4 s
%   Channel FrameRate 9.21 Hz -> 4.605 Hz per plane
%   512 x 440 px at 0.898 um/px, FOV 460 x 395 um
% Open field therefore has half the pain session's samples: expect fewer
% cells found and less power to separate a cell from background.
SESSIONS = { ...
  'pain',      'D:\20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_15-56-48'; ...
  'openfield', 'D:\20260911_CEANTSR1_65_15_openfield(100_50um)_555mi_2026-09-11_15-27-52' };
WHICH   = {'openfield'}; % {'pain'} | {'openfield'} | {'pain','openfield'}
PLANES  = {'A', 'B'};
DO_SPLIT   = true;       % skipped per session if plane_A/B.tif already there
DO_EXTRACT = true;

% ---------------- EXTRACT options ----------------
o = struct();
o.ca_frame_rate_hz = 4.605;            % 9.21 Hz / 2 ETL depths
o.extract_bin_time = 4;                % for CELL FINDING only
o.trace_on_unbinned = true;            % traces come back at 4.605 Hz
o.use_zscore_before_extract = false;   % real dF/F downstream
o.extract_preset = 'low_snr';
o.avg_cell_radius = 8;                 % measured median 7.9 px
o.n_frames_check = 100;
o.skip_video_01 = true;
o.skip_video_03 = true;
o.save_trace_files = true;

% Be a quiet neighbour: the other MATLAB holds a 31-worker pool.
try
    p = gcp('nocreate');
    if isempty(p)
        parpool('Processes', 4);
    end
catch ME
    warning('parpool unavailable (%s) - continuing single-threaded.', ...
            ME.message);
end

for s = 1:size(SESSIONS, 1)
    key = SESSIONS{s, 1};
    if ~any(strcmpi(key, WHICH)), continue; end
    sdir = SESSIONS{s, 2};
    odir = fullfile(sdir, 'output_split');
    if ~isfolder(odir), mkdir(odir); end

    fprintf('\n==================== %s ====================\n', upper(key));
    fprintf('  session: %s\n  output : %s\n', sdir, odir);

    pA = fullfile(odir, 'plane_A.tif');
    pB = fullfile(odir, 'plane_B.tif');
    if DO_SPLIT && ~(isfile(pA) && isfile(pB))
        pain_2plane_step1_split_planes(sdir, odir);
    else
        fprintf('  split skipped (plane_A/B.tif present)\n');
    end

    if ~DO_EXTRACT, continue; end
    for k = 1:numel(PLANES)
        pl = PLANES{k};
        tif = fullfile(odir, ['plane_', pl, '.tif']);
        if ~isfile(tif)
            warning('missing %s - skipped', tif);
            continue;
        end
        % one subfolder per plane: step2 writes several FIXED filenames
        % (analysis_results.mat, QC_report.mat, final_analysis_results.mat)
        % that would otherwise overwrite each other between planes
        pout = fullfile(odir, ['plane_', pl]);
        if ~isfolder(pout), mkdir(pout); end
        fprintf('\n----- EXTRACT: plane %s -----\n', pl);
        t0 = tic;
        try
            pain_2plane_step2_extract(tif, pout, o);
            fprintf('  plane %s done in %.0f s\n', pl, toc(t0));
        catch ME
            fprintf(2, '  plane %s FAILED: %s\n', pl, ME.message);
            for e = 1:numel(ME.stack)
                fprintf(2, '      %s line %d\n', ME.stack(e).name, ...
                        ME.stack(e).line);
            end
        end
    end
end

fprintf('\n==================== summary ====================\n');
for s = 1:size(SESSIONS, 1)
    key = SESSIONS{s, 1};
    if ~any(strcmpi(key, WHICH)), continue; end
    odir = fullfile(SESSIONS{s, 2}, 'output_split');
    for k = 1:numel(PLANES)
        f = fullfile(odir, ['plane_', PLANES{k}], ...
                     'final_analysis_results.mat');
        if isfile(f)
            w = whos('-file', f);
            nm = {w.name};
            if any(strcmp(nm, 'output'))
                S = load(f, 'output');
                nc = size(S.output.spatial_weights, 3);
                fr = 'binned';
                if isfield(S.output, 'traces_are_full_rate') && ...
                        S.output.traces_are_full_rate
                    fr = sprintf('%.3f Hz full rate', ...
                                 S.output.trace_frame_rate_hz);
                end
                fprintf('  %-10s plane %s : %3d cells, traces %s\n', ...
                        key, PLANES{k}, nc, fr);
            end
        else
            fprintf('  %-10s plane %s : no result\n', key, PLANES{k});
        end
    end
end
