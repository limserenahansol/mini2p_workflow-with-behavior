function pain_2plane_step5_merge(output_folder, opts)
% Merge neuron traces (from EXTRACT) with behavioral events into a single
% synchronized file at the neuron frame rate.
%
% Inputs (all expected in output_folder):
%   final_analysis_results.mat  (from Step 2: deltaF_over_F, output)
%   timestamps.mat              (from Step 3: neuron/behavior timestamps)
%   behavior_events_cam1.mat    (from Step 4: scored events)
%
% Output:
%   final_neuron_behavior.mat
%     .deltaF_over_F         (N_neuron_frames x N_cells)
%     .neuron_timestamps     (N_neuron_frames x 1 datetime)
%     .neuron_seconds        (N_neuron_frames x 1 double)
%     .event_vector          (N_neuron_frames x 1 logical)
%     .event_timestamps_original   (M x 1 datetime)
%     .spatial_weights       (H x W x N_cells)
%     .metadata              struct with rates, paths, etc.
%
% Usage:
%   pain_2plane_step5_merge(output_folder)
%   pain_2plane_step5_merge(output_folder, opts)
%     opts.behavior_file  - name of behavior events file (default: behavior_events_cam1.mat)

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'behavior_file')
    opts.behavior_file = 'behavior_events_cam1.mat';
end

disp('===== Step 5: Merge Neuron Traces + Behavior =====');

% ---- Load EXTRACT results ----
extract_file = fullfile(output_folder, 'final_analysis_results.mat');
if ~isfile(extract_file)
    error('EXTRACT results not found: %s', extract_file);
end
disp(['  Loading: ', extract_file]);
ex = load(extract_file);
deltaF_over_F = ex.deltaF_over_F;
spatial_weights = ex.output.spatial_weights;
n_neuron_frames = size(deltaF_over_F, 1);
n_cells = size(deltaF_over_F, 2);
disp(['    Neuron frames: ', num2str(n_neuron_frames), '  Cells: ', num2str(n_cells)]);

% ---- Load timestamps ----
ts_file = fullfile(output_folder, 'timestamps.mat');
if ~isfile(ts_file)
    error('Timestamps not found: %s', ts_file);
end
disp(['  Loading: ', ts_file]);
ts = load(ts_file);
neuron_timestamps = ts.neuron_timestamps;
neuron_seconds = ts.neuron_seconds;
neuron_frame_rate_hz = ts.neuron_frame_rate_hz;

if numel(neuron_timestamps) ~= n_neuron_frames
    warning('Neuron timestamp count (%d) != trace frames (%d). Trimming to shorter.', ...
        numel(neuron_timestamps), n_neuron_frames);
    n_use = min(numel(neuron_timestamps), n_neuron_frames);
    neuron_timestamps = neuron_timestamps(1:n_use);
    neuron_seconds = neuron_seconds(1:n_use);
    deltaF_over_F = deltaF_over_F(1:n_use, :);
    n_neuron_frames = n_use;
end

% ---- Load behavior events ----
behav_file = fullfile(output_folder, opts.behavior_file);
if ~isfile(behav_file)
    warning('Behavior events file not found: %s. Creating empty event vector.', behav_file);
    event_vector = false(n_neuron_frames, 1);
    event_timestamps_original = datetime.empty(0, 1);
else
    disp(['  Loading: ', behav_file]);
    bev = load(behav_file);
    event_frames_behav = bev.event_frames;
    event_timestamps_original = bev.event_timestamps;
    disp(['    Scored events: ', num2str(numel(event_frames_behav))]);

    % ---- Temporal alignment ----
    % Map behavioral event timestamps to neuron frame indices.
    % For each neuron frame i, check if any event falls within its time window.
    dt_neuron = 1 / neuron_frame_rate_hz;
    event_vector = false(n_neuron_frames, 1);

    if ~isempty(event_timestamps_original) && ~isempty(neuron_timestamps)
        event_sec = seconds(event_timestamps_original - neuron_timestamps(1));
        neuron_sec = seconds(neuron_timestamps - neuron_timestamps(1));

        for ei = 1:numel(event_sec)
            t_event = event_sec(ei);
            % Find the neuron frame whose center is closest
            [~, best_nf] = min(abs(neuron_sec - t_event));
            % Only assign if within half a neuron frame interval
            if abs(neuron_sec(best_nf) - t_event) <= dt_neuron / 2
                event_vector(best_nf) = true;
            else
                % Still assign to nearest if within one full interval
                if abs(neuron_sec(best_nf) - t_event) <= dt_neuron
                    event_vector(best_nf) = true;
                end
            end
        end
        disp(['    Events mapped to neuron frames: ', num2str(sum(event_vector))]);
    elseif ~isempty(event_frames_behav)
        % No timestamps available; use frame-rate ratio for alignment
        behav_rate = ts.behav_frame_rate_hz;
        ratio = behav_rate / neuron_frame_rate_hz;
        mapped_nf = round(double(event_frames_behav) / ratio);
        mapped_nf = mapped_nf(mapped_nf >= 1 & mapped_nf <= n_neuron_frames);
        event_vector(mapped_nf) = true;
        disp(['    Events mapped by rate ratio: ', num2str(sum(event_vector))]);
    end
end

% ---- Build metadata ----
metadata = struct();
metadata.neuron_frame_rate_hz = neuron_frame_rate_hz;
metadata.behav_frame_rate_hz = ts.behav_frame_rate_hz;
metadata.n_neuron_frames = n_neuron_frames;
metadata.n_cells = n_cells;
metadata.n_events = sum(event_vector);
metadata.behavior_file = opts.behavior_file;
metadata.created = datetime('now');

% ---- Save ----
out_file = fullfile(output_folder, 'final_neuron_behavior.mat');
save(out_file, 'deltaF_over_F', 'neuron_timestamps', 'neuron_seconds', ...
    'event_vector', 'event_timestamps_original', 'spatial_weights', 'metadata', '-v7.3');

disp(['  Saved: ', out_file]);
disp(['    deltaF_over_F:  ', num2str(n_neuron_frames), ' x ', num2str(n_cells)]);
disp(['    event_vector:   ', num2str(n_neuron_frames), ' x 1  (', num2str(sum(event_vector)), ' events)']);
disp('===== Step 5 complete =====');
end
