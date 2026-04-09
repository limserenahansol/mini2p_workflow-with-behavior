function results = downstream_step2_peri_event_traces(output_folder, opts)
% Compute and plot peri-event (time-locked) dF/F traces for each stimulus type.
%
% For every event type, extracts a window of dF/F around each event onset
% and plots the trial-averaged trace (mean +/- SEM across trials) for each
% cell, plus a population average.
%
% Required:
%   output_folder - folder containing final_neuron_behavior.mat
%
% Optional fields in opts:
%   .pre_sec         - seconds before event (default 3)
%   .post_sec        - seconds after event  (default 5)
%   .neuron_frame_rate - Hz (default 4.605)
%   .save_fig        - true to save figures as PNG (default true)
%   .event_types     - cell array of event types to analyze
%                      (default: all 4 stimuli + mouse_reaction)

if nargin < 2, opts = struct(); end

def = @(f,v) tern(isfield(opts,f), opts.(f), v);
pre_sec  = def('pre_sec', 3);
post_sec = def('post_sec', 5);
nfr      = def('neuron_frame_rate', 4.605);
save_fig = def('save_fig', true);

    function v = tern(c,a,b)
        if c, v = a; else, v = b; end
    end

disp('===== Downstream Step 2: Peri-Event dF/F Traces =====');

% ---- Load data ----
merged = load(fullfile(output_folder, 'final_neuron_behavior.mat'));
dff = merged.deltaF_over_F;
[n_frames, n_cells] = size(dff);

% Event definitions: {field_name, display_name, color}
all_events = {
    'soft_touch_vector',     'Soft Touch',      [0.2 0.8 0.2];
    'strong_touch_vector',   'Strong Touch',    [1.0 0.6 0.0];
    'mechanic_pain_vector',  'Mechanic Pain',   [1.0 0.2 0.2];
    'thermo_pain_vector',    'Thermo Pain',     [0.8 0.2 1.0];
    'mouse_reaction_vector', 'Mouse Reaction',  [0.2 0.6 1.0];
};

if isfield(opts, 'event_types')
    keep = ismember(all_events(:,1), opts.event_types);
    all_events = all_events(keep, :);
end

pre_frames  = round(pre_sec * nfr);
post_frames = round(post_sec * nfr);
window_len  = pre_frames + post_frames + 1;
time_axis   = (-pre_frames:post_frames) / nfr;

results = struct();
fig_dir = fullfile(output_folder, 'figures');
if save_fig && ~isfolder(fig_dir), mkdir(fig_dir); end

for ei = 1:size(all_events, 1)
    field = all_events{ei, 1};
    label = all_events{ei, 2};
    clr   = all_events{ei, 3};

    if ~isfield(merged, field)
        disp(['  Skipping ', label, ' (field not found)']);
        continue;
    end
    event_vec = merged.(field);
    event_idx = find(event_vec);

    % Remove events too close to edges
    event_idx(event_idx <= pre_frames) = [];
    event_idx(event_idx > n_frames - post_frames) = [];

    n_trials = numel(event_idx);
    disp(['  ', label, ': ', num2str(n_trials), ' trials']);

    if n_trials == 0
        continue;
    end

    % Extract peri-event traces: trials x time x cells
    peri = zeros(n_trials, window_len, n_cells);
    for ti = 1:n_trials
        win = event_idx(ti) + (-pre_frames:post_frames);
        peri(ti, :, :) = dff(win, :);
    end

    % Store results
    res.event_type  = field;
    res.label       = label;
    res.event_idx   = event_idx;
    res.n_trials    = n_trials;
    res.peri_traces = peri;
    res.time_axis   = time_axis;
    res.mean_trace  = squeeze(mean(peri, 1));
    if n_trials > 1
        res.sem_trace = squeeze(std(peri, 0, 1)) / sqrt(n_trials);
    else
        res.sem_trace = zeros(window_len, n_cells);
    end
    results.(matlab.lang.makeValidName(field)) = res;

    % ---- Plot: Population average ----
    pop_mean = mean(res.mean_trace, 2);
    pop_sem  = std(res.mean_trace, 0, 2) / sqrt(n_cells);

    fig1 = figure('Position', [100 100 800 400], 'Color', 'w');
    hold on;
    fill([time_axis, fliplr(time_axis)], ...
         [pop_mean'+pop_sem', fliplr(pop_mean'-pop_sem')], ...
         clr, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(time_axis, pop_mean, 'Color', clr, 'LineWidth', 2);
    xline(0, '--k', 'LineWidth', 1);
    xlabel('Time from event (s)');
    ylabel('\DeltaF/F');
    title(sprintf('%s — Population average (%d cells, %d trials)', label, n_cells, n_trials));
    set(gca, 'FontSize', 12);
    hold off;

    if save_fig
        saveas(fig1, fullfile(fig_dir, ['peri_event_pop_', field, '.png']));
    end

    % ---- Plot: Individual cell traces (tiled) ----
    n_per_page = min(n_cells, 12);
    n_pages = ceil(n_cells / n_per_page);
    for pg = 1:n_pages
        fig2 = figure('Position', [50 50 1200 800], 'Color', 'w');
        sgtitle(sprintf('%s — Per-Cell Peri-Event Traces (page %d/%d)', label, pg, n_pages), ...
            'FontSize', 14);
        cell_start = (pg-1)*n_per_page + 1;
        cell_end   = min(n_cells, pg*n_per_page);
        ncols = ceil(sqrt(n_per_page));
        nrows = ceil(n_per_page / ncols);

        for ci = cell_start:cell_end
            subplot(nrows, ncols, ci - cell_start + 1);
            hold on;
            mn = res.mean_trace(:, ci);
            se = res.sem_trace(:, ci);
            fill([time_axis, fliplr(time_axis)], ...
                 [mn'+se', fliplr(mn'-se')], ...
                 clr, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            plot(time_axis, mn, 'Color', clr, 'LineWidth', 1.5);
            xline(0, '--k');
            title(sprintf('Cell %d', ci), 'FontSize', 9);
            if ci == cell_start
                xlabel('Time (s)'); ylabel('\DeltaF/F');
            end
            set(gca, 'FontSize', 8);
            hold off;
        end

        if save_fig
            saveas(fig2, fullfile(fig_dir, sprintf('peri_event_cells_%s_p%d.png', field, pg)));
        end
    end
end

% ---- Save results ----
out_file = fullfile(output_folder, 'peri_event_results.mat');
save(out_file, 'results', 'time_axis', 'pre_sec', 'post_sec', 'nfr', '-v7.3');
disp(['  Saved: ', out_file]);
disp('===== Downstream Step 2 complete =====');

end
