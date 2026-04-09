% speed_dff_extract_HS_tiff_integrated_v2.m
% Integrated pipeline: best of your script + colleague's (batch, MC template, z-score, dual-channel option).
% Run mode: 'single' = one TIF; 'batch' = all TIFs in input_dir. Set paths and options below, then F5.
% EXTRACT: https://github.com/schnitzer-lab/EXTRACT-public

clear all;
close all;
set(0, 'RecursionLimit', 5000);

% ========== RUN MODE (from colleague: batch + platform) ==========
run_mode = 'single';   % 'single' = one file; 'batch' = multiple files in input_dir
% Batch only:
input_dir       = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\2p\20260320';
main_output_dir = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\2p\output_batch';
file_pattern    = '*.tif';   % e.g. 'm43*.tif' or '*.tif'

% ========== PLATFORM (from colleague: Mac/PC) ==========
if ispc
    EXTRACT_path   = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\EXTRACT-public';
    NoRMCorre_path = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\NoRMCorre';
    ActSort_path   = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\ActSort-public-main';
else
    EXTRACT_path   = '/path/to/EXTRACT-public';
    NoRMCorre_path = '/path/to/NoRMCorre';
    ActSort_path   = '/path/to/ActSort-public-main';
end
addpath(genpath(EXTRACT_path));
addpath(genpath(NoRMCorre_path));

% ========== SINGLE-FILE PATHS (used when run_mode == 'single') ==========
output_folder   = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\2p\multiscope_formini2p_ntsr0320_2';
file_path       = 'C:\Users\hsollim\Documents\MATLAB\MATLAB\2p\20260320';
file_name       = '2.5min_5989_1_00001.tif';

% ========== CHANNEL (from colleague: dual-channel option) ==========
dual_channel = false;   % true = load odd frames as green, even as red; EXTRACT runs on green only
use_red_for_mc_template = false;  % if dual_channel: use red for MC template (not implemented in v2; MC uses green)

% ========== SKIP FLAG ==========
skip_steps_1_2 = false;

% ========== MOTION CORRECTION (colleague: custom template from stable block) ==========
use_custom_mc_template = true;  % true = build template from frame_range, then run MC (better for difficult motion)
frame_range = 500:1500;          % frames used to find stable block (only if use_custom_mc_template)
templ_len   = 100;               % length of stable block for template
% NoRMCorre params (yours + colleague-style max_shift option)
mc_bin_width = 50;
mc_max_shift = 15;   % increase to 50 for large drift (colleague uses 50)
mc_us_fac    = 50;

% ========== PRE-EXTRACT (colleague: z-score + temporal bin for EXTRACT) ==========
use_zscore_before_extract = false;  % true = pixel-wise z-score before EXTRACT (standardizes intensity)
extract_bin_time = 4;              % I used 4 and Xiaochun used 8 1 = full rate; 8 = bin 8 frames for EXTRACT input (faster; traces at binned rate)

% ========== EXTRACT MODE ==========
extract_mode = 'final';   % 'step1' | 'step2' | 'final'

% ========== EXTRACT PRESET (yours = permissive/dim cells; colleague = stricter) ==========
extract_preset = 'permissive';   % 'permissive' (yours: dim/small cells) | 'stricter' (colleague: high SNR only)

% ========== OPTIONS ==========
run_actsort_post = false;
run_actsort_precompute = true;
run_signal_sorter_post = true;
% Trace/heatmap figures: 'extract' = all cells after EXTRACT+morph (matches yellow overlay).
% 'signal_sorter' = only cells SignalSorter accepted (often very few with automate=1 — that caused "138 cells but 2 traces").
dff_trace_source = 'extract';   % 'extract' | 'signal_sorter'
run_qc_report = true;
save_trace_trajectory_plots = true;
save_stacked_dff_traces = true;
save_trace_trajectory_per_cell = true;  % save one trace plot per cell (full-length raw; subfolder trace_trajectories_per_cell)
save_trace_files = false;   % (colleague-style) save Z and raw traces to MAT/CSV per session
trace_plot_n_cells = 30;
ca_frame_rate_hz = 30;
n_frames_check = 100; % set >0 to save quality-check videos: 01_input, 02_after_MC, 03_after_denoise, 04_EXTRACT_overlay. Set 0 to skip.
check_video_fps = 40; % default is 30 playback FPS for check AVIs (was 10 = choppy; match ca_frame_rate_hz for natural speed)
denoise_gauss_sigma = 1.5; % default is 1 spatial Gaussian on each frame after MC (larger = smoother movie but softer ROIs; try 1.2–1.5 for preview only)

fast_detect_mode = true;
detect_downsample_time_by = 4;
detect_downsample_space_by = 1;

% ========== BUILD FILE LIST ==========
if strcmpi(run_mode, 'batch')
    tif_list = dir(fullfile(input_dir, file_pattern));
    tif_list = tif_list(~[tif_list.isdir]);
    if isempty(tif_list)
        error('No files matching "%s" in %s', file_pattern, input_dir);
    end
    if ~isfolder(main_output_dir), mkdir(main_output_dir); end
    n_files = length(tif_list);
else
    n_files = 1;
    tif_list = [];
end

% ========== MAIN LOOP (batch or single) ==========
for file_idx = 1:n_files
    if strcmpi(run_mode, 'batch')
        file_name = tif_list(file_idx).name;
        full_file_path = fullfile(input_dir, file_name);
        [~, curr_header, ~] = fileparts(file_name);
        output_folder = fullfile(main_output_dir, curr_header);
        if ~isfolder(output_folder), mkdir(output_folder); end
    else
        full_file_path = fullfile(file_path, file_name);
        [~, curr_header, ~] = fileparts(file_name);
    end

    base_filename   = 'analysis_results.mat';
    output_mc_fname = fullfile(output_folder, [curr_header, '_1.h5']);
    extract_fname   = fullfile(output_folder, [curr_header, '_MC_extractout.mat']);
    ps_fname        = fullfile(output_folder, [curr_header, '_MC_extractout_ps.mat']);
    if ~isfolder(output_folder), mkdir(output_folder); end
    disp(['===== File ', num2str(file_idx), '/', num2str(n_files), ': ', curr_header, ' → ', output_folder]);

    write_check_video = @(mov, fname, nf) write_uint_video(mov, fullfile(output_folder, fname), nf, check_video_fps);

    %% ----- Load -----
    if skip_steps_1_2
        disp('skip_steps_1_2 = true → Loading from HDF5...');
        if ~isfile(output_mc_fname)
            error('HDF5 not found: %s', output_mc_fname);
        end
        denoised_data = h5read(output_mc_fname, '/mov');
        disp(['  Loaded size: ', mat2str(size(denoised_data))]);
    else
        if ~isfile(full_file_path)
            error('Input TIF not found: %s', full_file_path);
        end
        info = imfinfo(full_file_path);
        total_frames = numel(info);
        frame_height = info(1).Height;
        frame_width  = info(1).Width;

        if dual_channel
            frames_green = 1:2:total_frames;
            frames_red   = 2:2:total_frames;
            n_green = length(frames_green);
            n_red   = length(frames_red);
            frame_data_green = zeros(frame_height, frame_width, n_green, 'uint16');
            frame_data_red   = zeros(frame_height, frame_width, n_red,   'uint16');
            disp('Loading green (odd) and red (even) frames...');
            parfor (i = 1:n_green, 4)
                frame_data_green(:, :, i) = imread(full_file_path, 'Index', frames_green(i));
            end
            parfor (i = 1:n_red, 4)
                frame_data_red(:, :, i) = imread(full_file_path, 'Index', frames_red(i));
            end
            % Use green for main pipeline; red only for MC template if requested
            frame_data = frame_data_green;
            total_frames = n_green;
            clear frame_data_green frame_data_red;
        else
            frame_data = zeros(frame_height, frame_width, total_frames, 'uint16');
            disp('Loading all frames...');
            parfor (i = 1:total_frames, 4)
                frame_data(:, :, i) = imread(full_file_path, 'Index', i);
            end
        end

        downsampled_data = frame_data;
        clear frame_data;
        save(fullfile(output_folder, base_filename), 'total_frames', '-v7.3');

        if n_frames_check > 0
            write_check_video(downsampled_data, '01_check_input.avi', n_frames_check);
        end

        %% ----- Motion correction (with optional custom template) -----
        options_mc = NoRMCorreSetParms('d1', size(downsampled_data, 1), 'd2', size(downsampled_data, 2), ...
            'bin_width', mc_bin_width, 'max_shift', mc_max_shift, 'us_fac', mc_us_fac, 'iter', 1, 'output_type', 'mat');
        try
            gpuDevice();
            options_mc.gpu = true;
            disp('GPU: on');
        catch
            options_mc.gpu = false;
            disp('GPU: off');
        end

        mc_template = [];
        if use_custom_mc_template
            fr = frame_range;
            fr = fr(fr >= 1 & fr <= size(downsampled_data, 3));
            if length(fr) < templ_len
                warning('frame_range too short for template; using default MC.');
            else
                disp('Building MC template from stable block...');
                M_ini = single(downsampled_data(:, :, fr));
                if exist('spatial_bandpass', 'file') == 2
                    r_avg = 7;
                    M_bp = spatial_bandpass(M_ini, r_avg, 10, 2, 0);
                else
                    M_bp = M_ini;
                end
                % Simplified: use block with highest mean correlation (each frame vs block mean)
                nf = size(M_bp, 3);
                best_start = 1;
                best_score = -inf;
                for s = 1:min(50, nf - templ_len + 1)
                    block = M_bp(:, :, s:s + templ_len - 1);
                    mu = mean(block, 3);
                    sc = 0;
                    for t = 1:size(block, 3)
                        bt = block(:, :, t);
                        sc = sc + (mean(bt(:).*mu(:)) - mean(bt(:))*mean(mu(:))) / (eps + std(bt(:))*std(mu(:)));
                    end
                    sc = sc / size(block, 3);
                    if sc > best_score
                        best_score = sc;
                        best_start = s;
                    end
                end
                mc_template = mean(M_bp(:, :, best_start:best_start + templ_len - 1), 3);
                disp(['  Template from frames ', num2str(fr(best_start)), '-', num2str(fr(best_start + templ_len - 1))]);
            end
        end

        disp('Motion correction...');
        if isempty(mc_template)
            [motion_corrected_data, ~, ~] = normcorre_batch(downsampled_data, options_mc);
        else
            [motion_corrected_data, ~, ~] = normcorre_batch(downsampled_data, options_mc, mc_template);
        end
        clear downsampled_data;
        if n_frames_check > 0
            write_check_video(motion_corrected_data, '02_check_after_motion_correction.avi', n_frames_check);
        end

        disp('Denoising (spatial Gaussian)...');
        denoised_data = zeros(size(motion_corrected_data), 'single');
        for t = 1:size(motion_corrected_data, 3)
            denoised_data(:, :, t) = imgaussfilt(single(motion_corrected_data(:, :, t)), denoise_gauss_sigma);
        end
        save(fullfile(output_folder, base_filename), 'motion_corrected_data', '-append', '-v7.3');
        clear motion_corrected_data;

        if isfile(output_mc_fname), delete(output_mc_fname); end
        h5create(output_mc_fname, '/mov', size(denoised_data), 'Datatype', 'single');
        h5write(output_mc_fname, '/mov', single(denoised_data));

        if n_frames_check > 0
            write_check_video(denoised_data, '03_check_after_denoising.avi', n_frames_check);
        end
    end

    %% ----- Pre-EXTRACT: optional z-score and temporal bin -----
    M_for_extract = denoised_data;
    if use_zscore_before_extract
        disp('Z-scoring movie for EXTRACT...');
        mm = mean(M_for_extract, 3);
        ss = std(single(M_for_extract), 0, 3);
        ss(ss < 1e-6) = 1;
        M_for_extract = (single(M_for_extract) - mm) ./ ss;
    end
    if extract_bin_time > 1
        disp(['Temporal binning for EXTRACT: factor ', num2str(extract_bin_time)]);
        [h, w, T] = size(M_for_extract);
        Tb = floor(T / extract_bin_time);
        M_binned = zeros(h, w, Tb, 'single');
        for b = 1:Tb
            idx = (b-1)*extract_bin_time + (1:extract_bin_time);
            M_binned(:, :, b) = mean(M_for_extract(:, :, idx), 3);
        end
        M_for_extract = M_binned;
        clear M_binned;
    end

    % ----- EXTRACT -----
    avg_cell_radius = 8;
    if strcmpi(extract_preset, 'stricter')
        spatial_highpass_cutoff = 10;   % colleague-style
    else
        spatial_highpass_cutoff = 5;     % permissive for dim cells
    end
    M_proc = spatial_bandpass(M_for_extract, avg_cell_radius, spatial_highpass_cutoff, inf, 0);

    config = [];
    config = get_defaults(config);
    config.avg_cell_radius = avg_cell_radius;
    config.preprocess = 0;
    config.skip_dff = 1;
    config.F_per_pixel = ones(size(M_for_extract, 1), size(M_for_extract, 2));

    if strcmpi(extract_preset, 'stricter')
        config.cellfind_max_steps = 500;
        config.cellfind_min_snr   = 1;
        config.thresholds.T_min_snr = 10;
        config.max_iter = 10;
        config.num_partitions_x   = 1;
        config.num_partitions_y   = 1;
        config.spatial_highpass_cutoff = 10;
    else
        config.cellfind_max_steps = 3500;
        config.cellfind_min_snr   = 0.5;
        config.cellfind_numpix_threshold = 9;
        config.init_with_gaussian = true;
        config.thresholds.eccent_thresh = 7;
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
        config.num_partitions_x  = 2;
        config.num_partitions_y  = 2;
    end
    config.remove_background = true;

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
    clear M_proc M_for_extract;

    switch lower(extract_mode)
        case 'step1'
            config.max_iter = 0;
            config.visualize_cellfinding = 1;
            output = extractor(M_input, config);
            disp(['Candidates: ', num2str(size(output.spatial_weights, 3))]);
            clear M_input;
            continue;
        case 'step2'
            config.hyperparameter_tuning_flag = 1;
            output = extractor(M_input, config);
            if exist('plot_hyperparameter_curves', 'file') == 2
                fig_hp = figure;
                plot_hyperparameter_curves(output);
                saveas(fig_hp, fullfile(output_folder, [curr_header, '_hyperparam_curves.png']));
            end
            clear M_input;
            continue;
        otherwise
            output = extractor(M_input, config);
    end
    clear M_input;

    % ----- NaN removal + morphology cleanup (yours) -----
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

    [fov_h, fov_w, ~] = size(denoised_data);
    edge_margin = avg_cell_radius;
    min_area_px = round(pi * (avg_cell_radius * 0.45)^2);
    max_eccentricity = 0.95;
    min_solidity = 0.45;
    min_peak_dff = 0.05;
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
        if cr <= edge_margin || cr >= fov_h - edge_margin || cc_col <= edge_margin || cc_col >= fov_w - edge_margin
            continue;
        end
        tr = double(output.temporal_weights(:, i));
        f0 = prctile(tr, 20);
        f0_safe = max(abs(f0), prctile(abs(tr), 50));
        if f0_safe < 1, f0_safe = 1; end
        dff = (tr - f0) / f0_safe;
        peak_dff = prctile(dff, 99);
        keep(i) = props.Area >= min_area_px && props.Eccentricity <= max_eccentricity && ...
            props.Solidity >= min_solidity && peak_dff >= min_peak_dff;
    end
    output.temporal_weights = output.temporal_weights(:, keep);
    output.spatial_weights = output.spatial_weights(:, :, keep);

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

    if run_qc_report
        qc = build_qc_report(output, config, denoised_data, avg_cell_radius, spatial_highpass_cutoff, ...
            min_area_px, max_eccentricity, min_solidity, min_peak_dff, edge_margin, n_cells, numel(remove));
        save(fullfile(output_folder, 'QC_report.mat'), 'qc', '-v7.3');
        write_qc_text(qc, config, fullfile(output_folder, 'QC_report.txt'));
        write_qc_figure(qc, output, denoised_data, avg_cell_radius, n_final, fullfile(output_folder, 'QC_report.png'));
    end
    if n_frames_check > 0 && n_final > 0
        write_check_video_extract_overlay(denoised_data, output.spatial_weights, ...
            fullfile(output_folder, '04_check_EXTRACT_overlay.avi'), n_frames_check);
    end

    if run_actsort_post && run_actsort_precompute && isfile(output_mc_fname) && isfile(extract_fname)
        if isfolder(ActSort_path)
            addpath(genpath(ActSort_path));
            addpath(genpath(EXTRACT_path));
        end
        orig_dir = pwd;
        try
            cd(output_folder);
            PrecomputeCellCheck(extract_fname, output_mc_fname, 'parallel', false, ...
                'UIFigure', [], 'progressDlg', [], 'dt', 1, 'fast_features', true);
        catch ME
            disp(['ActSort Precompute failed: ', ME.message]);
        end
        cd(orig_dir);
    end

    % ----- SignalSorter -----
    if run_signal_sorter_post && isfile(ps_fname)
        load(ps_fname, 'output_ps');
    elseif run_signal_sorter_post
        try
            inputImages = output.spatial_weights;
            inputSignals = transpose(output.temporal_weights);
            iopts = [];
            iopts.inputMovie = output_mc_fname;
            iopts.inputDatasetName = '/mov';
            iopts.automate = 1;
            iopts.no_gui = 1;
            iopts.parallel = 0;
            [~, ~, choices] = ciapkg.classification.signalSorter(inputImages, inputSignals, 'options', iopts);
            numGood = sum(choices > 0);
            newImages = zeros(size(inputImages, 1), size(inputImages, 2), numGood);
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
        catch ME
            disp(['SignalSorter skipped: ', ME.message]);
            output_ps = output;
        end
    else
        output_ps = output;
    end

    n_extract = size(output.spatial_weights, 3);
    n_sorted  = size(output_ps.spatial_weights, 3);
    if run_signal_sorter_post && n_sorted < n_extract && strcmpi(dff_trace_source, 'signal_sorter')
        disp(['SignalSorter kept ', num2str(n_sorted), ' of ', num2str(n_extract), ' cells — ΔF/F traces match that subset only.']);
    elseif run_signal_sorter_post && n_sorted < n_extract && strcmpi(dff_trace_source, 'extract')
        disp(['SignalSorter kept ', num2str(n_sorted), ' of ', num2str(n_extract), ' cells; ΔF/F figures use all ', num2str(n_extract), ' EXTRACT cells (dff_trace_source=''extract'').']);
    end

    % ----- DeltaF/F and figures (use full-length raw traces, not binned) -----
    if strcmpi(dff_trace_source, 'signal_sorter')
        tw_out = output_ps;
    else
        tw_out = output;  % all morph-kept cells; matches cell_overlay_full_FOV
    end
    temporal_weights = tw_out.temporal_weights;
    T_full = size(denoised_data, 3);
    if extract_bin_time > 1 && size(temporal_weights, 1) < T_full
        % EXTRACT was run on binned data; project full-length denoised movie onto ROIs
        disp('Computing full-length traces (projecting raw denoised frames onto ROIs)...');
        [h, w, ~] = size(denoised_data);
        n_cells_tw = size(tw_out.spatial_weights, 3);
        M_flat = reshape(single(denoised_data), h*w, T_full);
        S_flat = reshape(tw_out.spatial_weights, h*w, n_cells_tw);
        temporal_weights = M_flat' * S_flat;  % (T_full x n_cells) full-length raw traces
    end
    F0 = mean(temporal_weights, 1);
    F0_safe = max(abs(F0), 1);
    deltaF_over_F = (temporal_weights - F0) ./ F0_safe;
    [max_peaks, ~] = max(deltaF_over_F, [], 1);
    [~, sorted_indices] = sort(max_peaks, 'descend');
    top_cells_indices = sorted_indices(1:min(50, length(sorted_indices)));

    fig1 = figure('Position', [100 100 800 800]);
    mean_img = mean(double(denoised_data), 3);
    imagesc(mean_img); colormap(gca, 'gray'); hold on;
    for i = 1:size(output.spatial_weights, 3)
        cell_mask = output.spatial_weights(:, :, i);
        bw = cell_mask > max(cell_mask(:)) * 0.15;
        B = bwboundaries(bw);
        for k = 1:length(B)
            plot(B{k}(:,2), B{k}(:,1), 'y', 'LineWidth', 1.5);
        end
    end
    hold off;
    title(sprintf('Detected cells: %d', size(output.spatial_weights, 3)));
    axis image;
    saveas(fig1, fullfile(output_folder, 'cell_overlay_full_FOV.png'));

    fig2 = figure;
    imagesc(deltaF_over_F'); colormap(gca, 'hot'); colorbar;
    xlabel('Frame'); ylabel('Cell ID'); title('\DeltaF/F heatmap');
    saveas(fig2, fullfile(output_folder, 'deltaF_over_F_heatmap.png'));

    if save_stacked_dff_traces && size(deltaF_over_F, 2) > 0
        if isempty(ca_frame_rate_hz) || ca_frame_rate_hz <= 0
            x = 1:size(deltaF_over_F, 1); xlab = 'Frame';
        else
            x = (0:size(deltaF_over_F, 1)-1) / ca_frame_rate_hz; xlab = 'Time (s)';
        end
        n_cells_plot = size(deltaF_over_F, 2);
        fig_r = figure('Position', [120 120 1200 800]);
        ax = axes(fig_r); hold(ax, 'on');
        dff_all = double(deltaF_over_F);
        span_all = max(prctile(dff_all(:), 99) - prctile(dff_all(:), 1), 1);
        y_offset = span_all * 1.2;
        for ci = 1:n_cells_plot
            plot(ax, x, dff_all(:, ci) + (ci-1)*y_offset, 'k', 'LineWidth', 0.8);
        end
        hold(ax, 'off');
        yticks(ax, (0:n_cells_plot-1)*y_offset);
        yticklabels(ax, arrayfun(@num2str, 1:n_cells_plot, 'UniformOutput', false));
        xlabel(ax, xlab); ylabel(ax, 'Cell ID'); title(ax, 'Stacked \DeltaF/F');
        grid(ax, 'on');
        saveas(fig_r, fullfile(output_folder, 'dff_traces_stacked_all_cells.png'));
    end

    if save_trace_trajectory_plots && size(deltaF_over_F, 2) > 0
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
    end

    % Per-cell trace trajectory (each single cell, full-length raw; no downsampling/binning)
    if save_trace_trajectory_per_cell && size(deltaF_over_F, 2) > 0
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

    if exist('plot_output_cellmap', 'file') == 2
        fig5 = figure;
        plot_output_cellmap(output, [], [], 'clim_scale', [0.2, 0.999]);
        saveas(fig5, fullfile(output_folder, 'extract_cellmap.png'));
    end

    save(fullfile(output_folder, base_filename), 'deltaF_over_F', 'max_peaks', 'top_cells_indices', '-append', '-v7.3');
    save(fullfile(output_folder, 'final_analysis_results.mat'), 'output', 'output_ps', 'deltaF_over_F', 'max_peaks', 'top_cells_indices', '-v7.3');

    % Colleague-style trace export (Z + raw) — same cells as deltaF_over_F / dff_trace_source
    if save_trace_files && size(deltaF_over_F, 2) > 0
        traces_raw = double(temporal_weights);
        traces_Z   = (traces_raw - mean(traces_raw, 1)) ./ max(std(traces_raw, 0, 1), 1e-10);
        save(fullfile(output_folder, [curr_header, '_traces_raw_Z.mat']), 'traces_raw', 'traces_Z', 'deltaF_over_F', '-v7.3');
        disp('  Saved: _traces_raw_Z.mat');
    end

    disp(['===== Done: ', curr_header, ' =====']);
end

disp('===== Pipeline finished =====');

% ===== LOCAL FUNCTIONS =====
function write_uint_video(mov, fname, nf, fps)
    if nargin < 4 || isempty(fps), fps = 10; end
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
    v.FrameRate = max(1, double(fps));
    open(v);
    for t = 1:nf
        fr = double(mov(:, :, t));
        fr = (fr - lo) / (hi - lo);
        fr = max(0, min(1, fr));
        writeVideo(v, repmat(fr, [1 1 3]));
    end
    close(v);
end

function write_check_video_extract_overlay(denoised, spatial_weights, fname, nf, fps)
    if nargin < 5 || isempty(fps), fps = 10; end
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
    v.FrameRate = max(1, double(fps));
    open(v);
    for t = 1:nf
        fr = double(denoised(:, :, t));
        fr = (fr - lo) / (hi - lo);
        fr = max(0, min(1, fr));
        frR = fr; frG = fr; frB = fr;
        frR(contour_mask) = 1; frG(contour_mask) = 1; frB(contour_mask) = 0;
        rgb = uint8(cat(3, frR, frG, frB) * 255);
        for li = 1:numel(label_info)
            pos = [label_info(li).col - 4, label_info(li).row - 5];
            rgb = insertText(rgb, pos, num2str(label_info(li).id), 'FontSize', 10, 'TextColor', 'cyan', 'BoxOpacity', 0);
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
    qc.cells = struct('id',{},'centroid_row',{},'centroid_col',{},'area',{},'eccentricity',{},'solidity',{},'peak_dff',{},'mean_trace',{},'snr',{});
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
        qc.cells(ci).id=ci; qc.cells(ci).centroid_row=pr.Centroid(2); qc.cells(ci).centroid_col=pr.Centroid(1);
        qc.cells(ci).area=pr.Area; qc.cells(ci).eccentricity=pr.Eccentricity; qc.cells(ci).solidity=pr.Solidity;
        qc.cells(ci).peak_dff=prctile(dff,99); qc.cells(ci).mean_trace=mean(tr); qc.cells(ci).snr=prctile(dff,99)/nb;
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
    if isfield(config, 'trace_output_option')
        L{end+1} = sprintf('  eccent=%.1f size=[%.2f %.1f]', config.thresholds.eccent_thresh, config.thresholds.size_lower_limit, config.thresholds.size_upper_limit);
    else
        L{end+1} = sprintf('  eccent=%.1f size=[%.2f %.1f]', config.thresholds.eccent_thresh, config.thresholds.size_lower_limit, config.thresholds.size_upper_limit);
    end
    L{end+1} = sprintf('Morph: area>=%d ecc<=%.2f sol>=%.2f dff>=%.3f edge=%d', ...
        qc.morph.min_area_px, qc.morph.max_ecc, qc.morph.min_sol, qc.morph.min_peak_dff, qc.morph.edge_margin);
    L{end+1} = sprintf('%4s %6s %6s %5s %5s %5s %8s %6s','ID','Row','Col','Area','Ecc','Sol','PkDFF','SNR');
    for ci=1:nf
        c=qc.cells(ci);
        L{end+1} = sprintf('%4d %6.1f %6.1f %5d %5.3f %5.3f %8.4f %6.2f', ...
            c.id,c.centroid_row,c.centroid_col,c.area,c.eccentricity,c.solidity,c.peak_dff,c.snr);
    end
    if nf>0
        L{end+1} = sprintf('Area: %d-%d  Ecc: %.3f-%.3f  DFF: %.4f-%.4f  SNR: %.2f-%.2f', ...
            min([qc.cells.area]),max([qc.cells.area]),min([qc.cells.eccentricity]),max([qc.cells.eccentricity]),...
            min([qc.cells.peak_dff]),max([qc.cells.peak_dff]),min([qc.cells.snr]),max([qc.cells.snr]));
    end
    for i=1:numel(L), fprintf(fid,'%s\n',L{i}); end
    fclose(fid);
end

function write_qc_figure(qc, output, denoised_data, avg_cell_radius, n_final, fname)
    qf = figure('Position',[50 50 1600 1000],'Color','w');
    subplot(2,3,1); imagesc(mean(double(denoised_data),3)); colormap(gca,'gray'); hold on;
    for ci=1:n_final
        cm=output.spatial_weights(:,:,ci); bw=cm>max(cm(:))*0.15;
        B=bwboundaries(bw); for k=1:length(B),plot(B{k}(:,2),B{k}(:,1),'y','LineWidth',1.2);end
        text(qc.cells(ci).centroid_col,qc.cells(ci).centroid_row,num2str(ci),'Color','c','FontSize',7,'FontWeight','bold','HorizontalAlignment','center');
    end; hold off; axis image; title(sprintf('Detected: %d',n_final));
    subplot(2,3,2); imagesc(max(double(denoised_data),[],3)); colormap(gca,'gray'); hold on;
    for ci=1:n_final
        cm=output.spatial_weights(:,:,ci); bw=cm>max(cm(:))*0.15;
        B=bwboundaries(bw); for k=1:length(B),plot(B{k}(:,2),B{k}(:,1),'r','LineWidth',1);end
    end; hold off; axis image; title('Max proj');
    subplot(2,3,3); imagesc(std(double(denoised_data),0,3)); colormap(gca,'hot'); colorbar; title('Std proj');
    subplot(2,3,4); if n_final>0
        scatter([qc.cells.area],[qc.cells.eccentricity],40,[qc.cells.peak_dff],'filled');
        colorbar; xlabel('Area'); ylabel('Ecc');
    end
    subplot(2,3,5); if n_final>0
        scatter([qc.cells.snr],[qc.cells.peak_dff],40,[qc.cells.solidity],'filled');
        colorbar; xlabel('SNR'); ylabel('PkDFF');
    end
    subplot(2,3,6); if n_final>0
        bar([qc.cells.area]); xlabel('Cell'); ylabel('Area');
    end
    sgtitle(sprintf('QC — %d cells',n_final),'FontSize',14,'FontWeight','bold');
    saveas(qf, fname);
end
