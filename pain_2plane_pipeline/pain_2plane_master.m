% pain_2plane_master.m
% Master script for the Pain 2-Plane Calcium Imaging + Behavior Pipeline.
% Set paths and skip flags below, then run (F5).
%
% Steps:
%   1. Load 2-plane interleaved TIFs -> max-project to single plane
%   2. Run EXTRACT (motion correction + cell detection + traces)
%   3. Parse TDMS timestamps (neuron + behavior cameras)
%   4. Manual behavioral video scoring (interactive GUI)
%   5. Merge neuron traces + behavior events into one file

clear; close all; clc;

% =====================================================================
%  PATHS — Edit these for your session
% =====================================================================

% Session folder (contains CellVideo1, MiceVideo1, MiceVideo2, etc.)
session_dir = 'C:\Users\hsollim\Downloads\20260409_Pain_2plane_-45+35+75%_28min_2026-04-08_17-48-09';

% Output folder (all intermediate and final results go here)
output_folder = fullfile(session_dir, 'output');

% Pipeline scripts folder (add to path)
pipeline_dir = fileparts(mfilename('fullpath'));
addpath(pipeline_dir);

% EXTRACT / NoRMCorre (added by step2, but ensure they're on path)
if ispc
    addpath(genpath('C:\Users\hsollim\Documents\MATLAB\MATLAB\EXTRACT-public'));
    addpath(genpath('C:\Users\hsollim\Documents\MATLAB\MATLAB\NoRMCorre'));
end

% =====================================================================
%  SKIP FLAGS — Set to true to skip completed steps
% =====================================================================

skip_step1 = false;   % Plane merge (only needed once)
skip_step2 = false;   % EXTRACT (re-run to tune params)
skip_step3 = false;   % Timestamp parsing (only needed once)
skip_step4 = false;   % Behavioral scoring (interactive; skip if already done)
skip_step5 = false;   % Merge (re-run after new scoring)

% Which camera to score in Step 4 (1 or 2)
score_camera_id = 1;

% =====================================================================
%  EXTRACT OPTIONS — Override any default from Step 2
% =====================================================================

extract_opts = struct();
extract_opts.ca_frame_rate_hz = 4.605;   % 9.21 Hz / 2 planes
extract_opts.extract_bin_time = 4;
extract_opts.extract_preset = 'permissive';
extract_opts.n_frames_check = 100;
extract_opts.skip_video_01 = true;       % skip input check video
extract_opts.skip_video_03 = true;       % skip denoising check video
% extract_opts.mc_max_shift = 15;
% extract_opts.avg_cell_radius = 8;

% =====================================================================
%  RUN PIPELINE
% =====================================================================

if ~isfolder(output_folder), mkdir(output_folder); end

combined_tif = fullfile(output_folder, 'combined_maxproj.tif');

% ----- Step 1: Merge planes -----
if ~skip_step1
    disp(' ');
    pain_2plane_step1_merge_planes(session_dir, output_folder);
else
    disp('Step 1 skipped (skip_step1 = true)');
end

% ----- Step 2: EXTRACT -----
if ~skip_step2
    if ~isfile(combined_tif)
        error('combined_maxproj.tif not found. Run Step 1 first.');
    end
    disp(' ');
    pain_2plane_step2_extract(combined_tif, output_folder, extract_opts);
else
    disp('Step 2 skipped (skip_step2 = true)');
end

% ----- Step 3: Parse timestamps -----
if ~skip_step3
    disp(' ');
    pain_2plane_step3_parse_timestamps(session_dir, output_folder);
else
    disp('Step 3 skipped (skip_step3 = true)');
end

% ----- Step 4: Behavioral scoring -----
if ~skip_step4
    disp(' ');
    ts_mat = fullfile(output_folder, 'timestamps.mat');
    if ~isfile(ts_mat)
        warning('timestamps.mat not found. Scoring without timestamps.');
        ts_mat = '';
    end
    pain_2plane_step4_behavior_scoring(session_dir, score_camera_id, output_folder, ts_mat);
else
    disp('Step 4 skipped (skip_step4 = true)');
end

% ----- Step 5: Merge -----
if ~skip_step5
    disp(' ');
    merge_opts = struct();
    merge_opts.behavior_file = sprintf('behavior_events_cam%d.mat', score_camera_id);
    pain_2plane_step5_merge(output_folder, merge_opts);
else
    disp('Step 5 skipped (skip_step5 = true)');
end

disp(' ');
disp('===== Pipeline complete =====');
disp(['  Output: ', output_folder]);
