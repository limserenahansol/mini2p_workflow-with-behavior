% speed_dff_extract_HS_tiff.m — One-run pipeline (F5 to run all)
% Set skip_steps_1_2 = true to re-run only EXTRACT (steps 3-5) using saved HDF5.
% EXTRACT: https://github.com/schnitzer-lab/EXTRACT-public

clear all;
close all;
set(0, 'RecursionLimit', 5000);

% ========== SKIP FLAG ==========
skip_steps_1_2 = false;

% ========== EXTRACT MODE (manual §5) ==========
%   'step1'  — cell finding only (max_iter=0, visualize on) for tuning
%   'step2'  — single refinement + hyperparameter curves for tuning
%   'final'  — full extraction (default, same speed as your reference code)
extract_mode = 'final';

% ========== OPTIONS ==========
run_actsort_post = false;
run_actsort_precompute = true;
run_signal_sorter_post = false;
run_qc_report = false;
save_trace_trajectory_plots = true;
save_stacked_dff_traces = true;
save_trace_trajectory_per_cell = true;  % save one trace plot per cell (full-length raw; subfolder trace_trajectories_per_cell)
trace_plot_n_cells = 30;
ca_frame_rate_hz = 30;
n_frames_check = 100;  % set >0 for quality-check videos: 01_after_downsampling, 02_after_MC, 03_after_denoise, 04_EXTRACT_overlay. Set 0 to skip.

% Fast ROI-detection mode (EXTRACT only). Other pipeline parts remain full-rate/full-size.
fast_detect_mode = true;
detect_downsample_time_by = 4;   % e.g., 30 Hz -> 7.5 Hz for detection/refinement
detect_downsample_space_by = 1;  % keep spatial resolution to preserve small ROIs

% ========== PATHS ==========
EXTRACT_path   = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\EXTRACT-public';
NoRMCorre_path = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\NoRMCorre';
ActSort_path   = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\ActSort-public-main';
addpath(genpath(EXTRACT_path));
addpath(genpath(NoRMCorre_path));

% ========== OUTPUT FOLDER AND FILE NAMES ==========
output_folder   = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\2p\multiscope_formini2p_2026March';
base_filename   = 'analysis_results.mat';
curr_header     = 'new_grinlens_2p_ceA';
output_mc_fname = fullfile(output_folder, [curr_header, '_1.h5']);
extract_fname   = fullfile(output_folder, [curr_header, '_MC_extractout.mat']);
ps_fname        = fullfile(output_folder, [curr_header, '_MC_extractout_ps.mat']);
if ~isfolder(output_folder), mkdir(output_folder); end
disp(['Results → ', output_folder]);

% ========== CHECK VIDEOS ==========
write_check_video = @(mov, fname, nf) write_uint_video(mov, fullfile(output_folder, fname), nf);

% ========== INPUT TIF ==========
file_path = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\2p\multiscope_formini2p_2026March';
file_name = '5949f_0.6_check_70insight3_00001.tif';
full_file_path = fullfile(file_path, file_name);

if skip_steps_1_2
    disp('skip_steps_1_2 = true → Loading denoised data from HDF5...');
    if ~isfile(output_mc_fname)
        error('HDF5 not found: %s\nRun with skip_steps_1_2 = false first.', output_mc_fname);
    end
    denoised_data = h5read(output_mc_fname, '/mov');
    disp(['  Loaded: ', output_mc_fname, '  size: ', mat2str(size(denoised_data))]);
else
    % ===== 1) LOAD ALL FRAMES AND TEMPORAL DOWNSAMPLE =====
    if ~isfile(full_file_path), error('Input TIF not found: %s', full_file_path); end
    info = imfinfo(full_file_path);
    total_frames = numel(info);
    frame_height = info(1).Height;
    frame_width  = info(1).Width;

    frame_data = zeros(frame_height, frame_width, total_frames, 'uint16');
    disp('Loading all frames...');
    parfor (i = 1:total_frames, 4)
        frame_data(:, :, i) = imread(full_file_path, 'Index', i);
    end

    downsample_factor_temporal = 1;
    num_ds_frames = floor(total_frames / downsample_factor_temporal);
    downsampled_data = zeros(frame_height, frame_width, num_ds_frames, 'uint16');
    disp('Temporal downsampling (no spatial)...');
    for i = 1:num_ds_frames
        sf = (i-1) * downsample_factor_temporal + 1;
        ef = sf + downsample_factor_temporal - 1;
        downsampled_data(:, :, i) = uint16(round(mean(double(frame_data(:, :, sf:ef)), 3)));
    end
    save(fullfile(output_folder, base_filename), 'total_frames', '-v7.3');
    clear frame_data;

    if n_frames_check > 0
        write_check_video(downsampled_data, '01_check_after_downsampling.avi', n_frames_check);
        disp('  Saved: 01_check_after_downsampling.avi');
    end

    % ===== 2) MOTION CORRECTION AND DENOISE =====
    options_mc = NoRMCorreSetParms('d1', size(downsampled_data, 1), 'd2', size(downsampled_data, 2), ...
        'bin_width', 50, 'max_shift', 15, 'us_fac', 50, 'iter', 1, 'output_type', 'mat');
    try gpuDevice(); options_mc.gpu = true; disp('GPU: on');
    catch, options_mc.gpu = false; disp('GPU: off'); end

    disp('Motion correction...');
    [motion_corrected_data, ~, ~] = normcorre_batch(downsampled_data, options_mc);
    clear downsampled_data;
    if n_frames_check > 0
        write_check_video(motion_corrected_data, '02_check_after_motion_correction.avi', n_frames_check);
        disp('  Saved: 02_check_after_motion_correction.avi');
    end

    disp('Denoising (spatial Gaussian only; preserve temporal dynamics)...');
    denoised_data = zeros(size(motion_corrected_data), 'single');
    for t = 1:size(motion_corrected_data, 3)
        denoised_data(:, :, t) = imgaussfilt(single(motion_corrected_data(:, :, t)), 1);
    end

    if isfile(output_mc_fname), delete(output_mc_fname); end
    h5create(output_mc_fname, '/mov', size(denoised_data), 'Datatype', 'single');
    h5write(output_mc_fname, '/mov', single(denoised_data));
    save(fullfile(output_folder, base_filename), 'motion_corrected_data', '-append', '-v7.3');
    clear motion_corrected_data;

    if n_frames_check > 0
        write_check_video(denoised_data, '03_check_after_denoising.avi', n_frames_check);
        disp('  Saved: 03_check_after_denoising.avi');
    end
end

% ===== 3) SPATIAL BANDPASS AND EXTRACT =====
avg_cell_radius = 8;
spatial_highpass_cutoff = 5;
M_proc = spatial_bandpass(denoised_data, avg_cell_radius, spatial_highpass_cutoff, inf, 0);

config = [];
config = get_defaults(config);
config.avg_cell_radius = avg_cell_radius;

config.preprocess = 0;
config.skip_dff = 1;
config.F_per_pixel = ones(size(denoised_data, 1), size(denoised_data, 2));

config.cellfind_max_steps = 3500;
config.cellfind_min_snr   = 0.5;
config.cellfind_numpix_threshold = 9;
config.init_with_gaussian = true;

config.thresholds.eccent_thresh    = 7;
config.smooth_S = true;
config.thresholds.spatial_corrupt_thresh = 1.5;

config.thresholds.size_upper_limit = 10;
config.thresholds.size_lower_limit = 0.15;

config.thresholds.T_min_snr = 3.0;
config.thresholds.low_ST_index_thresh = -0.05;
config.thresholds.low_ST_corr_thresh  = 0;

config.thresholds.S_dup_corr_thresh = 0.95;
config.thresholds.T_dup_corr_thresh = 0.95;

config.max_iter = 6;
config.remove_background = true;
config.num_partitions_x  = 2;
config.num_partitions_y  = 2;
if fast_detect_mode
    config.downsample_time_by = max(1, round(detect_downsample_time_by));
    config.downsample_space_by = max(1, round(detect_downsample_space_by));
else
    config.downsample_time_by = 1;
    config.downsample_space_by = 1;
end
config.use_gpu = 0;
config.visualize_cellfinding = 0;

M_input = M_proc;
clear M_proc;

disp(['EXTRACT detect mode: downsample_time_by=', num2str(config.downsample_time_by), ...
      ', downsample_space_by=', num2str(config.downsample_space_by)]);

switch lower(extract_mode)
    case 'step1'
        disp('=== STEP 1: Cell finding only (manual §5.1) ===');
        config.max_iter = 0;
        config.visualize_cellfinding = 1;
        output = extractor(M_input, config);
        disp(['Candidates found: ', num2str(size(output.spatial_weights, 3))]);
        disp('Tune cellfind_min_snr / T_min_snr, then set extract_mode=''step2''.');
        clear M_input;
        return;
    case 'step2'
        disp('=== STEP 2: Refinement tuning (manual §5.2) ===');
        config.hyperparameter_tuning_flag = 1;
        output = extractor(M_input, config);
        disp(['After 1 refinement: ', num2str(size(output.spatial_weights, 3)), ' cells.']);
        if exist('plot_hyperparameter_curves', 'file') == 2
            fig_hp = figure;
            plot_hyperparameter_curves(output);
            saveas(fig_hp, fullfile(output_folder, [curr_header, '_hyperparam_curves.png']));
            disp('Saved hyperparameter curves. Tune thresholds, then set extract_mode=''final''.');
        end
        clear M_input;
        return;
    otherwise
        disp('Running EXTRACT (final)...');
        output = extractor(M_input, config);
end
clear M_input;

% Remove NaN traces
n_cells  = size(output.temporal_weights, 2);
nf_trace = size(output.temporal_weights, 1);
remove = [];
for n = 1:n_cells
    if sum(isnan(output.temporal_weights(:, n))) == nf_trace
        remove(end+1) = n;
    end
end
for r = length(remove):-1:1
    output.temporal_weights(:, remove(r)) = [];
    output.spatial_weights(:, :, remove(r)) = [];
end

% Morphology + dynamics cleanup + edge rejection
[fov_h, fov_w, ~] = size(denoised_data);
edge_margin = avg_cell_radius;
min_area_px = round(pi * (avg_cell_radius * 0.45)^2);  % ~41 px — recover smaller dim neurons
max_eccentricity = 0.95;   % allow elongated soma-like ROIs
min_solidity = 0.45;       % keep dim cells with softer boundaries
min_peak_dff = 0.05;       % allow low-dynamic dim cells
keep = false(1, size(output.temporal_weights, 2));
for i = 1:size(output.temporal_weights, 2)
    cell_mask = output.spatial_weights(:, :, i);
    bw = cell_mask > max(cell_mask(:)) * 0.20;
    cc = bwconncomp(bw);
    if cc.NumObjects == 0, continue; end
    [~, idx] = max(cellfun(@numel, cc.PixelIdxList));
    bw_main = false(size(bw));
    bw_main(cc.PixelIdxList{idx}) = true;
    props = regionprops(bw_main, 'Area', 'Eccentricity', 'Solidity', 'Centroid');
    if isempty(props), continue; end

    cr = props.Centroid(2); cc_col = props.Centroid(1);
    if cr <= edge_margin || cr >= fov_h - edge_margin || ...
       cc_col <= edge_margin || cc_col >= fov_w - edge_margin
        continue;
    end

    tr = double(output.temporal_weights(:, i));
    f0 = prctile(tr, 20);
    f0_safe = max(abs(f0), prctile(abs(tr), 50));
    if f0_safe < 1, f0_safe = 1; end
    dff = (tr - f0) / f0_safe;
    peak_dff = prctile(dff, 99);
    keep(i) = props.Area >= min_area_px && ...
              props.Eccentricity <= max_eccentricity && ...
              props.Solidity >= min_solidity && ...
              peak_dff >= min_peak_dff;
end
output.temporal_weights = output.temporal_weights(:, keep);
output.spatial_weights = output.spatial_weights(:, :, keep);

% Force dense arrays (ActSort crashes on ndSparse)
if ~isa(output.spatial_weights, 'double')
    output.spatial_weights = full(double(output.spatial_weights));
end
if ~isa(output.temporal_weights, 'double')
    output.temporal_weights = full(double(output.temporal_weights));
end

save(extract_fname, 'output', '-v7.3');
save(fullfile(output_folder, base_filename), 'output', '-append', '-v7.3');
n_final = size(output.temporal_weights, 2);
disp(['EXTRACT done. Cells: ', num2str(n_final)]);

% ===== QC REPORT (optional; expensive) =====
if run_qc_report
    qc = build_qc_report(output, config, denoised_data, avg_cell_radius, spatial_highpass_cutoff, ...
        min_area_px, max_eccentricity, min_solidity, min_peak_dff, edge_margin, n_cells, numel(remove));
    save(fullfile(output_folder, 'QC_report.mat'), 'qc', '-v7.3');
    write_qc_text(qc, config, fullfile(output_folder, 'QC_report.txt'));
    write_qc_figure(qc, output, denoised_data, avg_cell_radius, n_final, fullfile(output_folder, 'QC_report.png'));
    disp('  Saved: QC_report.mat / .txt / .png');
end

if n_frames_check > 0 && size(output.spatial_weights, 3) > 0
    write_check_video_extract_overlay(denoised_data, output.spatial_weights, ...
        fullfile(output_folder, '04_check_after_EXTRACT_overlay.avi'), n_frames_check);
    disp('  Saved: 04_check_after_EXTRACT_overlay.avi');
end

% ===== 3b) ActSort precompute (manual-compatible dense MAT + H5) =====
if run_actsort_post && run_actsort_precompute && isfile(output_mc_fname) && isfile(extract_fname)
    disp('Running ActSort PrecomputeCellCheck...');
    if isfolder(ActSort_path)
        addpath(genpath(ActSort_path));
        addpath(genpath(EXTRACT_path));
    end
    orig_dir = pwd;
    try
        cd(output_folder);
        PrecomputeCellCheck(extract_fname, output_mc_fname, 'parallel', false, ...
            'UIFigure', [], 'progressDlg', [], 'dt', 1, 'fast_features', true);
        pc_files = dir(fullfile(output_folder, 'precomputed_*.mat'));
        if ~isempty(pc_files)
            disp(['  Precomputed file: ', pc_files(end).name]);
            disp('  Open CellSorter.mlapp → File → Open → select precomputed_*.mat');
        end
    catch ME
        disp(['ActSort PrecomputeCellCheck failed: ', ME.message]);
        disp('  Use ActSort Precompute manually: select EXTRACT .mat + H5 movie.');
    end
    cd(orig_dir);
end

% ===== 4) SIGNAL SORTER (optional; skips gracefully) =====
if run_signal_sorter_post && isfile(ps_fname)
    load(ps_fname, 'output_ps');
    disp('Loaded post-sorted output.');
elseif run_signal_sorter_post
    try
        inputImages  = output.spatial_weights;
        inputSignals = transpose(output.temporal_weights);
        iopts = [];
        iopts.inputMovie = output_mc_fname;
        iopts.inputDatasetName = '/mov';
        iopts.frameList = [];
        iopts.pixel_thresh = 200;
        iopts.ecc_thresh = 1.1;
        iopts.snr_thresh = 8;
        iopts.automate = 1;
        iopts.no_gui = 1;
        iopts.parallel = 0;
        if exist('ciapkg.api.manageParallelWorkers', 'file')
            ciapkg.api.manageParallelWorkers('parallel', false);
        end
        [~, ~, choices] = ciapkg.classification.signalSorter(inputImages, inputSignals, 'options', iopts);
        numGood = sum(choices > 0);
        newImages  = zeros(size(inputImages, 1), size(inputImages, 2), numGood);
        newSignals = zeros(numGood, size(inputSignals, 2));
        n = 0;
        for c = 1:numel(choices)
            if choices(c) > 0
                n = n + 1;
                newImages(:, :, n) = inputImages(:, :, c);
                newSignals(n, :) = inputSignals(c, :);
            end
        end
        output_ps = output;
        output_ps.spatial_weights = newImages;
        output_ps.temporal_weights = transpose(newSignals);
        save(ps_fname, 'output_ps', '-v7.3');
        disp('Signal sorter done.');
    catch ME
        disp(['Signal sorter skipped: ', ME.message]);
        output_ps = output;
    end
else
    output_ps = output;
end

% ===== 5) DELTA-F/F AND FIGURES =====
temporal_weights = output_ps.temporal_weights;
F0 = mean(temporal_weights, 1);
F0_safe = max(abs(F0), 1);
deltaF_over_F = (temporal_weights - F0) ./ F0_safe;

[max_peaks, ~] = max(deltaF_over_F, [], 1);
[~, sorted_indices] = sort(max_peaks, 'descend');
top_cells_indices = sorted_indices(1:min(50, length(sorted_indices)));

fig1 = figure('Position', [100 100 800 800]);
mean_img = mean(double(denoised_data), 3);
imagesc(mean_img); colormap(gca, 'gray'); hold on;
n_detected = size(output.spatial_weights, 3);
for i = 1:n_detected
    cell_mask = output.spatial_weights(:, :, i);
    bw = cell_mask > max(cell_mask(:)) * 0.15;
    B = bwboundaries(bw);
    for k = 1:length(B)
        plot(B{k}(:,2), B{k}(:,1), 'y', 'LineWidth', 1.5);
    end
end
hold off;
title(sprintf('Detected cells: %d (full FOV)', n_detected));
axis image; xlim([1 size(mean_img, 2)]); ylim([1 size(mean_img, 1)]);
saveas(fig1, fullfile(output_folder, 'cell_overlay_full_FOV.png'));

fig2 = figure;
imagesc(deltaF_over_F'); colormap(gca, 'hot'); colorbar;
xlabel('Frame'); ylabel('Cell ID'); title('\DeltaF/F heatmap');
saveas(fig2, fullfile(output_folder, 'deltaF_over_F_heatmap.png'));

% Stacked dF/F traces for all detected cells (row-style with Cell IDs)
if save_stacked_dff_traces && ~isempty(deltaF_over_F) && size(deltaF_over_F, 2) > 0
    if isempty(ca_frame_rate_hz) || ca_frame_rate_hz <= 0
        x = 1:size(deltaF_over_F, 1); xlab = 'Frame';
    else
        x = (0:size(deltaF_over_F, 1)-1) / ca_frame_rate_hz; xlab = 'Time (s)';
    end
    n_cells_plot = size(deltaF_over_F, 2);
    fig_rows = figure('Position', [120 120 1200 800]);
    ax_rows = axes(fig_rows); hold(ax_rows, 'on');
    dff_all = double(deltaF_over_F);
    span_all = prctile(dff_all(:), 99) - prctile(dff_all(:), 1);
    if span_all <= 0, span_all = 1; end
    y_offset = span_all * 1.2;
    for ci = 1:n_cells_plot
        y = dff_all(:, ci) + (ci-1) * y_offset;
        plot(ax_rows, x, y, 'k', 'LineWidth', 0.8);
    end
    hold(ax_rows, 'off');
    yticks(ax_rows, (0:n_cells_plot-1) * y_offset);
    yticklabels(ax_rows, arrayfun(@num2str, 1:n_cells_plot, 'UniformOutput', false));
    xlabel(ax_rows, xlab); ylabel(ax_rows, 'Cell ID (stacked)');
    title(ax_rows, sprintf('Stacked \\DeltaF/F traces (all %d cells)', n_cells_plot));
    grid(ax_rows, 'on');
    saveas(fig_rows, fullfile(output_folder, 'dff_traces_stacked_all_cells.png'));
    disp('  Saved: dff_traces_stacked_all_cells.png');
end

% Trace trajectories
if save_trace_trajectory_plots && ~isempty(deltaF_over_F) && size(deltaF_over_F, 2) > 0
    n_plot = min(trace_plot_n_cells, size(deltaF_over_F, 2));
    idx_plot = top_cells_indices(1:n_plot);
    if isempty(ca_frame_rate_hz) || ca_frame_rate_hz <= 0
        x = 1:size(deltaF_over_F, 1); xlab = 'Frame';
    else
        x = (0:size(deltaF_over_F, 1)-1) / ca_frame_rate_hz; xlab = 'Time (s)';
    end
    fig3 = figure('Position', [120 120 1100 450]);
    hold on;
    cmap = lines(max(n_plot, 7));
    for i = 1:n_plot
        plot(x, deltaF_over_F(:, idx_plot(i)), 'Color', cmap(i,:), 'LineWidth', 0.8);
    end
    hold off; xlabel(xlab); ylabel('\DeltaF/F');
    title(sprintf('Trace trajectories (top %d cells)', n_plot)); grid on;
    saveas(fig3, fullfile(output_folder, 'trace_trajectories_overlay.png'));

    fig4 = figure('Position', [120 120 1100 700]);
    hold on;
    dff_plot = deltaF_over_F(:, idx_plot);
    span = prctile(dff_plot(:), 99) - prctile(dff_plot(:), 1);
    if span <= 0, span = 1; end
    y_off = span * 1.2;
    for i = 1:n_plot
        plot(x, dff_plot(:, i) + (i-1) * y_off, 'k', 'LineWidth', 0.9);
    end
    hold off; xlabel(xlab); ylabel('Stacked \DeltaF/F');
    title(sprintf('Stacked traces (top %d cells)', n_plot)); grid on;
    saveas(fig4, fullfile(output_folder, 'trace_trajectories_stacked.png'));
    disp('  Saved: trace trajectory plots');
end

% Per-cell trace trajectory (each single cell, full-length raw)
if save_trace_trajectory_per_cell && ~isempty(deltaF_over_F) && size(deltaF_over_F, 2) > 0
    per_cell_dir = fullfile(output_folder, 'trace_trajectories_per_cell');
    if ~isfolder(per_cell_dir), mkdir(per_cell_dir); end
    if isempty(ca_frame_rate_hz) || ca_frame_rate_hz <= 0
        x = 1:size(deltaF_over_F, 1); xlab = 'Frame';
    else
        x = (0:size(deltaF_over_F, 1)-1) / ca_frame_rate_hz; xlab = 'Time (s)';
    end
    n_cells_all = size(deltaF_over_F, 2);
    for ci = 1:n_cells_all
        fig_c = figure('Position', [100 100 1000 280], 'Visible', 'off');
        plot(x, deltaF_over_F(:, ci), 'k', 'LineWidth', 0.8);
        xlabel(xlab); ylabel('\DeltaF/F');
        title(sprintf('Cell %d (full-length raw trace)', ci)); grid on;
        saveas(fig_c, fullfile(per_cell_dir, sprintf('cell_%03d.png', ci)));
        close(fig_c);
    end
    disp(['  Saved: ', num2str(n_cells_all), ' per-cell trace plots in trace_trajectories_per_cell/']);
end

% EXTRACT native cell map (manual §5.3)
if exist('plot_output_cellmap', 'file') == 2
    fig5 = figure;
    plot_output_cellmap(output, [], [], 'clim_scale', [0.2, 0.999]);
    saveas(fig5, fullfile(output_folder, 'extract_cellmap.png'));
    disp('  Saved: extract_cellmap.png');
end

save(fullfile(output_folder, base_filename), 'deltaF_over_F', 'max_peaks', 'top_cells_indices', '-append', '-v7.3');
save(fullfile(output_folder, 'final_analysis_results.mat'), 'output', 'output_ps', 'deltaF_over_F', 'max_peaks', 'top_cells_indices', '-v7.3');

disp('===== Pipeline finished =====');

% ===== LOCAL HELPER FUNCTIONS =====

function write_uint_video(mov, fname, nf)
    nf = min(nf, size(mov, 3));
    if nf < 1, return; end
    sample_idx = round(linspace(1, nf, min(20, nf)));
    all_vals = [];
    for s = sample_idx
        tmp = double(mov(:, :, s));
        all_vals = [all_vals; tmp(:)];
    end
    lo = prctile(all_vals, 1);
    hi = prctile(all_vals, 99.5);
    if hi <= lo, hi = lo + 1; end
    v = VideoWriter(fname, 'Uncompressed AVI');
    v.FrameRate = 10;
    open(v);
    for t = 1:nf
        fr = double(mov(:, :, t));
        fr = (fr - lo) / (hi - lo);
        fr = max(0, min(1, fr));
        writeVideo(v, repmat(fr, [1 1 3]));
    end
    close(v);
end

function write_check_video_extract_overlay(denoised, spatial_weights, fname, nf)
    nf = min(nf, size(denoised, 3));
    if nf < 1, return; end
    [h, w, ~] = size(denoised);
    n_cells = size(spatial_weights, 3);
    contour_mask = false(h, w);
    label_info = struct('row', {}, 'col', {}, 'id', {});
    for i = 1:n_cells
        cell_mask = spatial_weights(:, :, i);
        bw = cell_mask > max(cell_mask(:)) * 0.15;
        B = bwboundaries(bw);
        for k = 1:length(B)
            boundary = B{k};
            for p = 1:size(boundary, 1)
                r = boundary(p, 1); c = boundary(p, 2);
                for dr = -1:1
                    for dc = -1:1
                        rr = r + dr; cc = c + dc;
                        if rr >= 1 && rr <= h && cc >= 1 && cc <= w
                            contour_mask(rr, cc) = true;
                        end
                    end
                end
            end
        end
        props = regionprops(bw, 'Centroid');
        if ~isempty(props)
            label_info(end+1).row = round(props(1).Centroid(2));
            label_info(end).col = round(props(1).Centroid(1));
            label_info(end).id = i;
        end
    end
    sample_idx = round(linspace(1, nf, min(20, nf)));
    all_vals = [];
    for s = sample_idx
        tmp = double(denoised(:, :, s));
        all_vals = [all_vals; tmp(:)];
    end
    lo = prctile(all_vals, 1);
    hi = prctile(all_vals, 99.5);
    if hi <= lo, hi = lo + 1; end
    v = VideoWriter(fname, 'Uncompressed AVI');
    v.FrameRate = 10;
    open(v);
    for t = 1:nf
        fr = double(denoised(:, :, t));
        fr = (fr - lo) / (hi - lo);
        fr = max(0, min(1, fr));
        frR = fr; frG = fr; frB = fr;
        frR(contour_mask) = 1;
        frG(contour_mask) = 1;
        frB(contour_mask) = 0;
        rgb = cat(3, frR, frG, frB);
        rgb = uint8(rgb * 255);
        for li = 1:numel(label_info)
            pos = [label_info(li).col - 4, label_info(li).row - 5];
            rgb = insertText(rgb, pos, num2str(label_info(li).id), ...
                'FontSize', 10, 'TextColor', 'cyan', 'BoxOpacity', 0);
        end
        writeVideo(v, rgb);
    end
    close(v);
end

function qc = build_qc_report(output, config, denoised_data, avg_cell_radius, ...
    spatial_highpass_cutoff, min_area_px, max_ecc, min_sol, min_peak_dff, edge_margin, n_raw, n_removed)
    n_final = size(output.temporal_weights, 2);
    qc.n_cells_raw = n_raw;
    qc.n_cells_after_nan_removal = n_raw - n_removed;
    qc.n_cells_after_morphology = n_final;
    qc.config = config;
    qc.avg_cell_radius = avg_cell_radius;
    qc.spatial_highpass_cutoff = spatial_highpass_cutoff;
    qc.morph.min_area_px = min_area_px;
    qc.morph.max_ecc = max_ecc;
    qc.morph.min_sol = min_sol;
    qc.morph.min_peak_dff = min_peak_dff;
    qc.morph.edge_margin = edge_margin;
    qc.cells = struct('id',{},'centroid_row',{},'centroid_col',{},...
        'area',{},'eccentricity',{},'solidity',{},'peak_dff',{},'mean_trace',{},'snr',{});
    for ci = 1:n_final
        cm = output.spatial_weights(:,:,ci);
        bw = cm > max(cm(:))*0.20;
        cc = bwconncomp(bw);
        if cc.NumObjects==0
            pr = struct('Area',0,'Eccentricity',0,'Solidity',0,'Centroid',[0 0]);
        else
            [~,ix]=max(cellfun(@numel,cc.PixelIdxList));
            bm=false(size(bw)); bm(cc.PixelIdxList{ix})=true;
            pr=regionprops(bm,'Area','Eccentricity','Solidity','Centroid'); pr=pr(1);
        end
        tr=double(output.temporal_weights(:,ci));
        f0=prctile(tr,20); fs=max(abs(f0),prctile(abs(tr),50)); if fs<1,fs=1;end
        dff=(tr-f0)/fs;
        nb=prctile(dff,75)-prctile(dff,25); if nb<1e-3,nb=1e-3;end
        qc.cells(ci).id=ci; qc.cells(ci).centroid_row=pr.Centroid(2);
        qc.cells(ci).centroid_col=pr.Centroid(1); qc.cells(ci).area=pr.Area;
        qc.cells(ci).eccentricity=pr.Eccentricity; qc.cells(ci).solidity=pr.Solidity;
        qc.cells(ci).peak_dff=prctile(dff,99); qc.cells(ci).mean_trace=mean(tr);
        qc.cells(ci).snr=prctile(dff,99)/nb;
    end
end

function write_qc_text(qc, config, fname)
    fid = fopen(fname, 'w');
    nf = qc.n_cells_after_morphology;
    L = {};
    L{end+1} = sprintf('===== QC REPORT =====');
    L{end+1} = sprintf('Cells: raw=%d  after_NaN=%d  after_morph=%d', qc.n_cells_raw, qc.n_cells_after_nan_removal, nf);
    L{end+1} = sprintf('EXTRACT: radius=%d hp=%d steps=%d min_snr=%.2f T_snr=%.2f iter=%d', ...
        config.avg_cell_radius, qc.spatial_highpass_cutoff, config.cellfind_max_steps, ...
        config.cellfind_min_snr, config.thresholds.T_min_snr, config.max_iter);
    L{end+1} = sprintf('  eccent=%.1f corrupt=%.2f size=[%.2f %.1f] trace=%s', ...
        config.thresholds.eccent_thresh, config.thresholds.spatial_corrupt_thresh, ...
        config.thresholds.size_lower_limit, config.thresholds.size_upper_limit, config.trace_output_option);
    L{end+1} = sprintf('Morph: area>=%d ecc<=%.2f sol>=%.2f dff>=%.3f edge=%d', ...
        qc.morph.min_area_px, qc.morph.max_ecc, qc.morph.min_sol, qc.morph.min_peak_dff, qc.morph.edge_margin);
    L{end+1} = sprintf('%4s %6s %6s %5s %5s %5s %8s %6s','ID','Row','Col','Area','Ecc','Sol','PkDFF','SNR');
    for ci=1:nf
        c=qc.cells(ci);
        L{end+1} = sprintf('%4d %6.1f %6.1f %5d %5.3f %5.3f %8.4f %6.2f', ...
            c.id,c.centroid_row,c.centroid_col,c.area,c.eccentricity,c.solidity,c.peak_dff,c.snr);
    end
    if nf>0
        L{end+1} = sprintf('Area: %d-%d  Ecc: %.3f-%.3f  Sol: %.3f-%.3f  DFF: %.4f-%.4f  SNR: %.2f-%.2f', ...
            min([qc.cells.area]),max([qc.cells.area]),min([qc.cells.eccentricity]),max([qc.cells.eccentricity]),...
            min([qc.cells.solidity]),max([qc.cells.solidity]),min([qc.cells.peak_dff]),max([qc.cells.peak_dff]),...
            min([qc.cells.snr]),max([qc.cells.snr]));
    end
    for i=1:numel(L), disp(L{i}); fprintf(fid,'%s\n',L{i}); end
    fclose(fid);
end

function write_qc_figure(qc, output, denoised_data, avg_cell_radius, n_final, fname)
    qf = figure('Position',[50 50 1600 1000],'Color','w');
    subplot(2,3,1); imagesc(mean(double(denoised_data),3)); colormap(gca,'gray'); hold on;
    for ci=1:n_final
        cm=output.spatial_weights(:,:,ci); bw=cm>max(cm(:))*0.15;
        B=bwboundaries(bw); for k=1:length(B),plot(B{k}(:,2),B{k}(:,1),'y','LineWidth',1.2);end
        text(qc.cells(ci).centroid_col,qc.cells(ci).centroid_row,num2str(ci),...
            'Color','c','FontSize',7,'FontWeight','bold','HorizontalAlignment','center');
    end; hold off; axis image; title(sprintf('Detected: %d',n_final));
    subplot(2,3,2); imagesc(max(double(denoised_data),[],3)); colormap(gca,'gray'); hold on;
    for ci=1:n_final
        cm=output.spatial_weights(:,:,ci); bw=cm>max(cm(:))*0.15;
        B=bwboundaries(bw); for k=1:length(B),plot(B{k}(:,2),B{k}(:,1),'r','LineWidth',1);end
    end; hold off; axis image; title('Max proj + contours');
    subplot(2,3,3); imagesc(std(double(denoised_data),0,3)); colormap(gca,'hot'); hold on;
    for ci=1:n_final
        cm=output.spatial_weights(:,:,ci); bw=cm>max(cm(:))*0.15;
        B=bwboundaries(bw); for k=1:length(B),plot(B{k}(:,2),B{k}(:,1),'c','LineWidth',1);end
    end; hold off; axis image; title('Std proj + contours'); colorbar;
    subplot(2,3,4); if n_final>0
        scatter([qc.cells.area],[qc.cells.eccentricity],40,[qc.cells.peak_dff],'filled');
        colorbar; xlabel('Area'); ylabel('Ecc'); title('Area vs Ecc');
        for ci=1:n_final,text(double(qc.cells(ci).area)+0.5,qc.cells(ci).eccentricity,num2str(ci),'FontSize',6);end
    end
    subplot(2,3,5); if n_final>0
        scatter([qc.cells.snr],[qc.cells.peak_dff],40,[qc.cells.solidity],'filled');
        colorbar; xlabel('SNR'); ylabel('PkDFF'); title('SNR vs DFF');
        for ci=1:n_final,text(qc.cells(ci).snr+0.1,qc.cells(ci).peak_dff,num2str(ci),'FontSize',6);end
    end
    subplot(2,3,6); nt=min(n_final,36);
    if nt>0
        nc=ceil(sqrt(nt)); nr=ceil(nt/nc); tsz=2*avg_cell_radius+6;
        tile=zeros(nr*(tsz+1),nc*(tsz+1));
        for ci=1:nt
            cm=output.spatial_weights(:,:,ci);
            cr=round(qc.cells(ci).centroid_row); ccc=round(qc.cells(ci).centroid_col);
            r1=max(1,cr-avg_cell_radius-2); r2=min(size(cm,1),cr+avg_cell_radius+3);
            c1=max(1,ccc-avg_cell_radius-2); c2=min(size(cm,2),ccc+avg_cell_radius+3);
            p=imresize(cm(r1:r2,c1:c2),[tsz tsz]);
            ri=floor((ci-1)/nc); cii=mod(ci-1,nc);
            tile(ri*(tsz+1)+1:ri*(tsz+1)+tsz, cii*(tsz+1)+1:cii*(tsz+1)+tsz)=p;
        end
        imagesc(tile); colormap(gca,'hot'); axis image off; title(sprintf('Thumbnails 1-%d',nt));
    end
    sgtitle(sprintf('QC — %d cells',n_final),'FontSize',14,'FontWeight','bold');
    saveas(qf, fname);
end
