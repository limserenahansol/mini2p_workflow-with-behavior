function downstream_step3_auc_stats(output_folder, opts)
% Compute AUC (area under curve) of dF/F before vs after each event and
% generate bar graphs with paired statistics.
%
% For each event type and each cell:
%   AUC_pre  = integral of dF/F over [-2, 0] s relative to event
%   AUC_post = integral of dF/F over [0, +2] s relative to event
%
% Across trials, a paired Wilcoxon signed-rank test compares pre vs post.
%
% Required:
%   output_folder - folder containing peri_event_results.mat (from Step 2)
%
% Optional fields in opts:
%   .pre_window_sec    - seconds before event for AUC (default 2)
%   .post_window_sec   - seconds after event for AUC  (default 2)
%   .neuron_frame_rate - Hz (default 4.605)
%   .alpha             - significance level (default 0.05)
%   .save_fig          - save figures as PNG (default true)

if nargin < 2, opts = struct(); end

def = @(f,v) tern(isfield(opts,f), opts.(f), v);
pre_win  = def('pre_window_sec', 2);
post_win = def('post_window_sec', 2);
nfr      = def('neuron_frame_rate', 4.605);
alpha    = def('alpha', 0.05);
save_fig = def('save_fig', true);

    function v = tern(c,a,b)
        if c, v = a; else, v = b; end
    end

disp('===== Downstream Step 3: AUC Pre/Post Statistics =====');

% ---- Load peri-event results ----
peri_file = fullfile(output_folder, 'peri_event_results.mat');
if ~isfile(peri_file)
    error('peri_event_results.mat not found. Run downstream_step2 first.');
end
dat = load(peri_file);
results = dat.results;
time_axis = dat.time_axis;

fig_dir = fullfile(output_folder, 'figures');
if save_fig && ~isfolder(fig_dir), mkdir(fig_dir); end

event_fields = fieldnames(results);
auc_summary = struct();

% Event display colors
color_map = struct( ...
    'soft_touch_vector',     [0.2 0.8 0.2], ...
    'strong_touch_vector',   [1.0 0.6 0.0], ...
    'mechanic_pain_vector',  [1.0 0.2 0.2], ...
    'thermo_pain_vector',    [0.8 0.2 1.0], ...
    'mouse_reaction_vector', [0.2 0.6 1.0]);

for ei = 1:numel(event_fields)
    ef = event_fields{ei};
    res = results.(ef);
    label = res.label;
    n_trials = res.n_trials;
    n_cells = size(res.peri_traces, 3);

    if n_trials < 2
        disp(['  Skipping ', label, ' (< 2 trials)']);
        continue;
    end

    disp(['  ', label, ': ', num2str(n_trials), ' trials, ', num2str(n_cells), ' cells']);

    % Time indices for pre and post windows
    pre_idx  = time_axis >= -pre_win & time_axis < 0;
    post_idx = time_axis >= 0 & time_axis <= post_win;
    dt = 1 / nfr;

    % AUC per trial per cell: trapz approximation
    auc_pre  = zeros(n_trials, n_cells);
    auc_post = zeros(n_trials, n_cells);
    for ci = 1:n_cells
        for ti = 1:n_trials
            trace = squeeze(res.peri_traces(ti, :, ci));
            auc_pre(ti, ci)  = trapz(time_axis(pre_idx),  trace(pre_idx));
            auc_post(ti, ci) = trapz(time_axis(post_idx), trace(post_idx));
        end
    end

    % Trial-averaged AUC per cell
    mean_auc_pre  = mean(auc_pre, 1);
    mean_auc_post = mean(auc_post, 1);

    % Paired test per cell (across trials)
    p_per_cell = ones(1, n_cells);
    for ci = 1:n_cells
        if n_trials >= 5
            p_per_cell(ci) = signrank(auc_pre(:,ci), auc_post(:,ci));
        else
            [~, p_per_cell(ci)] = ttest(auc_pre(:,ci), auc_post(:,ci));
        end
    end

    % Population-level test (average across cells for each trial)
    pop_pre  = mean(auc_pre, 2);
    pop_post = mean(auc_post, 2);
    if n_trials >= 5
        p_pop = signrank(pop_pre, pop_post);
    else
        [~, p_pop] = ttest(pop_pre, pop_post);
    end

    % Store
    s.event_type    = res.event_type;
    s.label         = label;
    s.n_trials      = n_trials;
    s.auc_pre       = auc_pre;
    s.auc_post      = auc_post;
    s.mean_auc_pre  = mean_auc_pre;
    s.mean_auc_post = mean_auc_post;
    s.p_per_cell    = p_per_cell;
    s.p_population  = p_pop;
    auc_summary.(ef) = s;

    % ---- Figure 1: Population bar graph ----
    if isfield(color_map, res.event_type)
        clr = color_map.(res.event_type);
    else
        clr = [0.5 0.5 0.5];
    end

    fig1 = figure('Position', [100 100 500 450], 'Color', 'w');
    bar_data = [mean(pop_pre), mean(pop_post)];
    bar_err  = [std(pop_pre)/sqrt(n_trials), std(pop_post)/sqrt(n_trials)];
    b = bar(1:2, bar_data, 0.6); hold on;
    b.FaceColor = 'flat';
    b.CData(1,:) = [0.6 0.6 0.6];
    b.CData(2,:) = clr;
    errorbar(1:2, bar_data, bar_err, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);

    % Significance asterisks
    max_y = max(bar_data + bar_err) * 1.15;
    line([1, 2], [max_y, max_y], 'Color', 'k', 'LineWidth', 1);
    if p_pop < 0.001
        sig_str = '***';
    elseif p_pop < 0.01
        sig_str = '**';
    elseif p_pop < alpha
        sig_str = '*';
    else
        sig_str = 'n.s.';
    end
    text(1.5, max_y * 1.05, sprintf('%s (p=%.4f)', sig_str, p_pop), ...
        'HorizontalAlignment', 'center', 'FontSize', 12);

    set(gca, 'XTick', [1, 2], 'XTickLabel', ...
        {sprintf('Pre (-%ds)', pre_win), sprintf('Post (+%ds)', post_win)}, ...
        'FontSize', 12);
    ylabel('AUC of \DeltaF/F');
    title(sprintf('%s — AUC Pre vs Post (%d trials)', label, n_trials));
    hold off;

    if save_fig
        saveas(fig1, fullfile(fig_dir, ['auc_population_', res.event_type, '.png']));
    end

    % ---- Figure 2: Per-cell bar graph ----
    fig2 = figure('Position', [100 100 max(600, n_cells*60), 450], 'Color', 'w');
    x = 1:n_cells;
    bar_w = 0.35;
    hold on;
    bar(x - bar_w/2, mean_auc_pre, bar_w, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'none');
    bar(x + bar_w/2, mean_auc_post, bar_w, 'FaceColor', clr, 'EdgeColor', 'none');

    for ci = 1:n_cells
        if p_per_cell(ci) < alpha
            max_val = max(mean_auc_pre(ci), mean_auc_post(ci));
            text(ci, max_val * 1.1, '*', 'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'Color', 'r');
        end
    end

    xlabel('Cell #'); ylabel('AUC of \DeltaF/F');
    title(sprintf('%s — Per-Cell AUC (gray=pre, color=post)', label));
    legend({'Pre-event', 'Post-event'}, 'Location', 'best');
    set(gca, 'XTick', x, 'FontSize', 10);
    hold off;

    if save_fig
        saveas(fig2, fullfile(fig_dir, ['auc_per_cell_', res.event_type, '.png']));
    end

    disp(['    Population p = ', num2str(p_pop, '%.4f'), '  ', sig_str]);
    n_sig = sum(p_per_cell < alpha);
    disp(['    Significant cells: ', num2str(n_sig), '/', num2str(n_cells)]);
end

% ---- Save ----
out_file = fullfile(output_folder, 'auc_statistics.mat');
save(out_file, 'auc_summary', 'pre_win', 'post_win', 'alpha', '-v7.3');
disp(['  Saved: ', out_file]);
disp('===== Downstream Step 3 complete =====');

end
