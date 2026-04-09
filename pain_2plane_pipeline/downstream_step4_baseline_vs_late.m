function downstream_step4_baseline_vs_late(output_folder, opts)
% Compare neuronal activity between an early baseline period and a late
% period to assess sensitization / habituation over the session.
%
% Default periods:
%   Baseline = first 5 minutes  (0–300 s)
%   Late     = minutes 23–28    (1380–1680 s)
%
% Metrics per cell:
%   - Mean dF/F
%   - Event rate (transients/min detected by threshold crossing)
%   - Peak dF/F
%
% Statistical tests: paired Wilcoxon signed-rank (or t-test if n_cells < 5).
%
% Required:
%   output_folder - folder containing final_neuron_behavior.mat
%
% Optional fields in opts:
%   .baseline_start_sec  - start of baseline (default 0)
%   .baseline_end_sec    - end of baseline   (default 300 = 5 min)
%   .late_start_sec      - start of late period (default 1380 = 23 min)
%   .late_end_sec        - end of late period   (default 1680 = 28 min)
%   .neuron_frame_rate   - Hz (default 4.605)
%   .transient_thresh_sd - SD threshold for transient detection (default 2)
%   .save_fig            - save figures (default true)

if nargin < 2, opts = struct(); end

def = @(f,v) tern(isfield(opts,f), opts.(f), v);
bl_start = def('baseline_start_sec', 0);
bl_end   = def('baseline_end_sec', 300);
lt_start = def('late_start_sec', 1380);
lt_end   = def('late_end_sec', 1680);
nfr      = def('neuron_frame_rate', 4.605);
thresh_sd = def('transient_thresh_sd', 2);
save_fig = def('save_fig', true);

    function v = tern(c,a,b)
        if c, v = a; else, v = b; end
    end

disp('===== Downstream Step 4: Baseline vs Late Comparison =====');

% ---- Load data ----
merged = load(fullfile(output_folder, 'final_neuron_behavior.mat'));
dff = merged.deltaF_over_F;
[n_frames, n_cells] = size(dff);
time_sec = (0:n_frames-1) / nfr;

% Frame ranges
bl_frames = find(time_sec >= bl_start & time_sec < bl_end);
lt_frames = find(time_sec >= lt_start & time_sec < lt_end);

if isempty(bl_frames)
    error('No frames in baseline window [%.0f, %.0f] s', bl_start, bl_end);
end
if isempty(lt_frames)
    error('No frames in late window [%.0f, %.0f] s. Total recording = %.1f s', ...
        lt_start, lt_end, time_sec(end));
end

disp(sprintf('  Baseline: %.0f–%.0f s  (%d frames)', bl_start, bl_end, numel(bl_frames)));
disp(sprintf('  Late:     %.0f–%.0f s  (%d frames)', lt_start, lt_end, numel(lt_frames)));
disp(sprintf('  Cells: %d', n_cells));

bl_dur_min = (bl_end - bl_start) / 60;
lt_dur_min = (lt_end - lt_start) / 60;

% ---- Compute metrics per cell ----
mean_dff_bl  = zeros(n_cells, 1);
mean_dff_lt  = zeros(n_cells, 1);
peak_dff_bl  = zeros(n_cells, 1);
peak_dff_lt  = zeros(n_cells, 1);
event_rate_bl = zeros(n_cells, 1);
event_rate_lt = zeros(n_cells, 1);

for ci = 1:n_cells
    trace_bl = dff(bl_frames, ci);
    trace_lt = dff(lt_frames, ci);

    mean_dff_bl(ci) = mean(trace_bl);
    mean_dff_lt(ci) = mean(trace_lt);
    peak_dff_bl(ci) = max(trace_bl);
    peak_dff_lt(ci) = max(trace_lt);

    % Transient detection: threshold = median + thresh_sd * MAD
    full_trace = dff(:, ci);
    med_val = median(full_trace);
    mad_val = mad(full_trace, 1) * 1.4826;
    thr = med_val + thresh_sd * mad_val;

    above_bl = trace_bl > thr;
    above_lt = trace_lt > thr;
    n_trans_bl = sum(diff([0; above_bl]) == 1);
    n_trans_lt = sum(diff([0; above_lt]) == 1);
    event_rate_bl(ci) = n_trans_bl / bl_dur_min;
    event_rate_lt(ci) = n_trans_lt / lt_dur_min;
end

% ---- Statistical tests ----
if n_cells >= 5
    test_name = 'Wilcoxon signed-rank';
    p_mean = signrank(mean_dff_bl, mean_dff_lt);
    p_peak = signrank(peak_dff_bl, peak_dff_lt);
    p_rate = signrank(event_rate_bl, event_rate_lt);
else
    test_name = 'Paired t-test';
    [~, p_mean] = ttest(mean_dff_bl, mean_dff_lt);
    [~, p_peak] = ttest(peak_dff_bl, peak_dff_lt);
    [~, p_rate] = ttest(event_rate_bl, event_rate_lt);
end

disp(['  Test: ', test_name]);
disp(sprintf('    Mean dF/F:     p = %.4f', p_mean));
disp(sprintf('    Peak dF/F:     p = %.4f', p_peak));
disp(sprintf('    Event rate:    p = %.4f', p_rate));

fig_dir = fullfile(output_folder, 'figures');
if save_fig && ~isfolder(fig_dir), mkdir(fig_dir); end

% ---- Figure 1: Mean dF/F bar graph ----
fig1 = figure('Position', [100 100 500 450], 'Color', 'w');
plot_paired_bar(mean_dff_bl, mean_dff_lt, p_mean, ...
    'Mean \DeltaF/F', ...
    sprintf('Baseline (0–%d min)', bl_end/60), ...
    sprintf('Late (%d–%d min)', lt_start/60, lt_end/60), ...
    [0.4 0.7 1.0], [1.0 0.4 0.4]);
if save_fig, saveas(fig1, fullfile(fig_dir, 'baseline_vs_late_mean_dff.png')); end

% ---- Figure 2: Peak dF/F bar graph ----
fig2 = figure('Position', [100 100 500 450], 'Color', 'w');
plot_paired_bar(peak_dff_bl, peak_dff_lt, p_peak, ...
    'Peak \DeltaF/F', ...
    sprintf('Baseline (0–%d min)', bl_end/60), ...
    sprintf('Late (%d–%d min)', lt_start/60, lt_end/60), ...
    [0.4 0.7 1.0], [1.0 0.4 0.4]);
if save_fig, saveas(fig2, fullfile(fig_dir, 'baseline_vs_late_peak_dff.png')); end

% ---- Figure 3: Event rate bar graph ----
fig3 = figure('Position', [100 100 500 450], 'Color', 'w');
plot_paired_bar(event_rate_bl, event_rate_lt, p_rate, ...
    'Transient rate (events/min)', ...
    sprintf('Baseline (0–%d min)', bl_end/60), ...
    sprintf('Late (%d–%d min)', lt_start/60, lt_end/60), ...
    [0.4 0.7 1.0], [1.0 0.4 0.4]);
if save_fig, saveas(fig3, fullfile(fig_dir, 'baseline_vs_late_event_rate.png')); end

% ---- Figure 4: Per-cell scatter (baseline vs late mean dF/F) ----
fig4 = figure('Position', [100 100 500 450], 'Color', 'w');
hold on;
scatter(mean_dff_bl, mean_dff_lt, 60, 'filled', 'MarkerFaceColor', [0.3 0.3 0.8]);
lims = [min([mean_dff_bl; mean_dff_lt])*0.9, max([mean_dff_bl; mean_dff_lt])*1.1];
plot(lims, lims, '--k', 'LineWidth', 1);
xlabel(sprintf('Baseline Mean \\DeltaF/F (0–%d min)', bl_end/60));
ylabel(sprintf('Late Mean \\DeltaF/F (%d–%d min)', lt_start/60, lt_end/60));
title(sprintf('Per-Cell Mean \\DeltaF/F  (p=%.4f, %s)', p_mean, test_name));
for ci = 1:n_cells
    text(mean_dff_bl(ci)+0.002, mean_dff_lt(ci), num2str(ci), 'FontSize', 8);
end
set(gca, 'FontSize', 11);
hold off;
if save_fig, saveas(fig4, fullfile(fig_dir, 'baseline_vs_late_scatter.png')); end

% ---- Figure 5: Full session trace (population mean) ----
fig5 = figure('Position', [50 50 1400 350], 'Color', 'w');
pop_trace = mean(dff, 2);
plot(time_sec / 60, pop_trace, 'Color', [0.2 0.2 0.2], 'LineWidth', 0.5);
hold on;
% Shade baseline and late windows
yl = ylim;
fill([bl_start bl_end bl_end bl_start]/60, [yl(1) yl(1) yl(2) yl(2)], ...
    [0.4 0.7 1.0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill([lt_start lt_end lt_end lt_start]/60, [yl(1) yl(1) yl(2) yl(2)], ...
    [1.0 0.4 0.4], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
xlabel('Time (min)'); ylabel('Population mean \DeltaF/F');
title('Full Session — Blue = Baseline, Red = Late period');
set(gca, 'FontSize', 11);
hold off;
if save_fig, saveas(fig5, fullfile(fig_dir, 'baseline_vs_late_session_trace.png')); end

% ---- Save results ----
bl_vs_late = struct();
bl_vs_late.mean_dff_baseline  = mean_dff_bl;
bl_vs_late.mean_dff_late      = mean_dff_lt;
bl_vs_late.peak_dff_baseline  = peak_dff_bl;
bl_vs_late.peak_dff_late      = peak_dff_lt;
bl_vs_late.event_rate_baseline = event_rate_bl;
bl_vs_late.event_rate_late    = event_rate_lt;
bl_vs_late.p_mean_dff  = p_mean;
bl_vs_late.p_peak_dff  = p_peak;
bl_vs_late.p_event_rate = p_rate;
bl_vs_late.test_used    = test_name;
bl_vs_late.baseline_sec = [bl_start, bl_end];
bl_vs_late.late_sec     = [lt_start, lt_end];
bl_vs_late.n_cells      = n_cells;

out_file = fullfile(output_folder, 'baseline_vs_late_stats.mat');
save(out_file, 'bl_vs_late', '-v7.3');
disp(['  Saved: ', out_file]);
disp('===== Downstream Step 4 complete =====');

end

% =========================================================================
function plot_paired_bar(vals_bl, vals_lt, p_val, y_label, bl_name, lt_name, clr_bl, clr_lt)
    n = numel(vals_bl);
    hold on;

    % Individual paired lines
    for ci = 1:n
        plot([1, 2], [vals_bl(ci), vals_lt(ci)], '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
    end

    % Bar means
    mn = [mean(vals_bl), mean(vals_lt)];
    se = [std(vals_bl)/sqrt(n), std(vals_lt)/sqrt(n)];
    b = bar(1:2, mn, 0.5);
    b.FaceColor = 'flat';
    b.CData(1,:) = clr_bl;
    b.CData(2,:) = clr_lt;
    b.FaceAlpha = 0.6;
    errorbar(1:2, mn, se, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);

    % Scatter individual cells
    scatter(ones(n,1), vals_bl, 30, clr_bl, 'filled', 'MarkerFaceAlpha', 0.7);
    scatter(2*ones(n,1), vals_lt, 30, clr_lt, 'filled', 'MarkerFaceAlpha', 0.7);

    % Significance
    max_y = max([vals_bl; vals_lt]) * 1.2;
    line([1, 2], [max_y, max_y], 'Color', 'k', 'LineWidth', 1);
    if p_val < 0.001
        sig = '***';
    elseif p_val < 0.01
        sig = '**';
    elseif p_val < 0.05
        sig = '*';
    else
        sig = 'n.s.';
    end
    text(1.5, max_y * 1.06, sprintf('%s (p=%.4f)', sig, p_val), ...
        'HorizontalAlignment', 'center', 'FontSize', 12);

    set(gca, 'XTick', [1, 2], 'XTickLabel', {bl_name, lt_name}, 'FontSize', 11);
    ylabel(y_label);
    title(sprintf('%s — %s vs %s (n=%d cells)', y_label, bl_name, lt_name, n));
    hold off;
end
