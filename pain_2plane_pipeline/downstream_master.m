% downstream_master.m
% Master script for the Downstream Analysis Pipeline.
% Runs AFTER the main pipeline (Steps 1-5) has completed.
%
% Prerequisites:
%   - final_neuron_behavior.mat (from pain_2plane_step5_merge)
%   - Behavioral camera AVI folders (MiceVideo1, MiceVideo2) for Step 1
%
% Steps:
%   DS-1. Synchronized 3-view video clip (neuron dF/F + cam1 + cam2)
%   DS-2. Peri-event (time-locked) dF/F traces per stimulus type
%   DS-3. AUC pre/post event statistics with bar graphs
%   DS-4. Baseline (0-5 min) vs Late (23-28 min) activity comparison

clear; close all; clc;

% =====================================================================
%  PATHS — Same as main pipeline
% =====================================================================

session_dir   = 'C:\Users\hsollim\Downloads\20260409_Pain_2plane_-45+35+75%_28min_2026-04-08_17-48-09';
output_folder = fullfile(session_dir, 'output');

pipeline_dir = fileparts(mfilename('fullpath'));
addpath(pipeline_dir);

% =====================================================================
%  SKIP FLAGS — Set to true to skip completed steps
% =====================================================================

skip_ds1 = false;   % Synchronized 3-view video
skip_ds2 = false;   % Peri-event traces
skip_ds3 = false;   % AUC statistics
skip_ds4 = false;   % Baseline vs Late comparison

% =====================================================================
%  DS-1: Synchronized 3-View Video Clip
% =====================================================================
% Generates a composite video: cam1 | cam2 on top, dF/F heatmap on bottom
% Shows a 3-minute window starting from neuron frame 4400, at 5x speed.

if ~skip_ds1
    disp(' ');
    vid_opts = struct();
    vid_opts.start_neuron_frame = 4400;     % start frame (~16 min into session)
    vid_opts.clip_duration_sec  = 180;      % 3 minutes of real time
    vid_opts.speed_multiplier   = 5;        % 5x playback
    vid_opts.output_fps         = 30;
    vid_opts.output_filename    = 'sync_3view_clip.avi';
    vid_opts.cam1_subfolder     = 'MiceVideo1';
    vid_opts.cam2_subfolder     = 'MiceVideo2';
    vid_opts.neuron_frame_rate  = 4.605;
    vid_opts.behav_frame_rate   = 30;
    vid_opts.video_width        = 1280;
    vid_opts.video_height       = 720;

    downstream_step1_sync_video(session_dir, output_folder, vid_opts);
else
    disp('DS-1 skipped (skip_ds1 = true)');
end

% =====================================================================
%  DS-2: Peri-Event (Time-Locked) dF/F Traces
% =====================================================================
% For each stimulus type, extracts a [-3, +5] s window around each event
% and plots trial-averaged traces (mean +/- SEM) per cell and population.

if ~skip_ds2
    disp(' ');
    peri_opts = struct();
    peri_opts.pre_sec  = 3;                 % 3 s before event
    peri_opts.post_sec = 5;                 % 5 s after event
    peri_opts.neuron_frame_rate = 4.605;
    peri_opts.save_fig = true;

    downstream_step2_peri_event_traces(output_folder, peri_opts);
else
    disp('DS-2 skipped (skip_ds2 = true)');
end

% =====================================================================
%  DS-3: AUC Pre/Post Event Statistics
% =====================================================================
% Computes AUC of dF/F in [-2, 0] s vs [0, +2] s around each event.
% Bar graphs with Wilcoxon signed-rank tests per cell and population.

if ~skip_ds3
    disp(' ');
    auc_opts = struct();
    auc_opts.pre_window_sec    = 2;         % 2 s before event
    auc_opts.post_window_sec   = 2;         % 2 s after event
    auc_opts.neuron_frame_rate = 4.605;
    auc_opts.alpha             = 0.05;
    auc_opts.save_fig          = true;

    downstream_step3_auc_stats(output_folder, auc_opts);
else
    disp('DS-3 skipped (skip_ds3 = true)');
end

% =====================================================================
%  DS-4: Baseline vs Late Activity Comparison
% =====================================================================
% Compares first 5 min (baseline) with minutes 23-28 (late period).
% Metrics: mean dF/F, peak dF/F, transient rate. Paired stats.

if ~skip_ds4
    disp(' ');
    bl_opts = struct();
    bl_opts.baseline_start_sec = 0;
    bl_opts.baseline_end_sec   = 300;       % 0-5 min
    bl_opts.late_start_sec     = 1380;      % 23 min
    bl_opts.late_end_sec       = 1680;      % 28 min
    bl_opts.neuron_frame_rate  = 4.605;
    bl_opts.transient_thresh_sd = 2;
    bl_opts.save_fig           = true;

    downstream_step4_baseline_vs_late(output_folder, bl_opts);
else
    disp('DS-4 skipped (skip_ds4 = true)');
end

disp(' ');
disp('===== Downstream Pipeline complete =====');
disp(['  Output: ', output_folder]);
disp(['  Figures: ', fullfile(output_folder, 'figures')]);
