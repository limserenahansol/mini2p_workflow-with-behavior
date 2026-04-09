function pain_2plane_step5_merge(output_folder, opts)
% Merge neuron traces (from EXTRACT) with behavioral events into a single
% synchronized file at the neuron frame rate.
%
% Supports two cameras with distinct event types:
%   Camera 1 → "Soft_touch" / "Strong_touch" / "Mechanic_pain" / "Thermo_pain"
%   Camera 2 → "Mouse_reaction" / "Reaction_offset"
%
% Inputs (all expected in output_folder):
%   final_analysis_results.mat  (from Step 2)
%   timestamps.mat              (from Step 3)
%   behavior_events_cam1.mat    (from Step 4, camera 1)
%   behavior_events_cam2.mat    (from Step 4, camera 2) — optional
%
% Output:
%   final_neuron_behavior.mat
%     .deltaF_over_F              (N x N_cells)
%     .neuron_timestamps          (N x 1 datetime)
%     .neuron_seconds             (N x 1 double)
%     .soft_touch_vector          (N x 1 logical) — cam1: soft touch
%     .strong_touch_vector        (N x 1 logical) — cam1: strong touch
%     .mechanic_pain_vector       (N x 1 logical) — cam1: mechanical pain
%     .thermo_pain_vector         (N x 1 logical) — cam1: thermal pain
%     .mouse_reaction_vector      (N x 1 logical) — cam2 primary events
%     .reaction_offset_vector     (N x 1 logical) — cam2 secondary events
%     .event_vector               (N x 1 logical) — any event (backward compat)
%     .event_timestamps_original  (struct with cam1/cam2 fields)
%     .spatial_weights            (H x W x N_cells)
%     .metadata                   struct
%
% Usage:
%   pain_2plane_step5_merge(output_folder)
%   pain_2plane_step5_merge(output_folder, opts)
%     opts.merge_both_cameras  - true (default) to load both cam1 and cam2

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'merge_both_cameras'), opts.merge_both_cameras = true; end
if ~isfield(opts, 'behavior_file'), opts.behavior_file = 'behavior_events_cam1.mat'; end

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

dt_neuron = 1 / neuron_frame_rate_hz;

% ---- Helper: map events to neuron frames ----
    function vec = map_events_to_neuron(bev_file, label_filter)
        vec = false(n_neuron_frames, 1);
        if ~isfile(bev_file)
            return;
        end
        bev = load(bev_file);
        if ~isfield(bev, 'event_frames') || isempty(bev.event_frames)
            return;
        end

        % Filter by label if specified
        if nargin >= 2 && ~isempty(label_filter) && isfield(bev, 'event_labels')
            mask = strcmp(bev.event_labels, label_filter);
            sel_frames = bev.event_frames(mask);
            if isfield(bev, 'event_timestamps') && ~isempty(bev.event_timestamps)
                sel_ts = bev.event_timestamps(mask);
            else
                sel_ts = [];
            end
        else
            sel_frames = bev.event_frames;
            if isfield(bev, 'event_timestamps')
                sel_ts = bev.event_timestamps;
            else
                sel_ts = [];
            end
        end

        if isempty(sel_frames), return; end

        if ~isempty(sel_ts) && ~isempty(neuron_timestamps)
            event_sec = seconds(sel_ts - neuron_timestamps(1));
            neuron_sec = seconds(neuron_timestamps - neuron_timestamps(1));
            for ei = 1:numel(event_sec)
                [~, best_nf] = min(abs(neuron_sec - event_sec(ei)));
                if abs(neuron_sec(best_nf) - event_sec(ei)) <= dt_neuron
                    vec(best_nf) = true;
                end
            end
        else
            behav_rate = ts.behav_frame_rate_hz;
            ratio = behav_rate / neuron_frame_rate_hz;
            mapped_nf = round(double(sel_frames) / ratio);
            mapped_nf = mapped_nf(mapped_nf >= 1 & mapped_nf <= n_neuron_frames);
            vec(mapped_nf) = true;
        end
    end

% ---- Process Camera 1 events (4 stimuli types) ----
cam1_file = fullfile(output_folder, 'behavior_events_cam1.mat');
soft_touch_vector     = map_events_to_neuron(cam1_file, 'Soft_touch');
strong_touch_vector   = map_events_to_neuron(cam1_file, 'Strong_touch');
mechanic_pain_vector  = map_events_to_neuron(cam1_file, 'Mechanic_pain');
thermo_pain_vector    = map_events_to_neuron(cam1_file, 'Thermo_pain');

% Backward compatibility: old-style 'A' or 'Stimuli_given' → soft_touch
if sum(soft_touch_vector) == 0 && isfile(cam1_file)
    soft_touch_vector = soft_touch_vector | map_events_to_neuron(cam1_file, 'Stimuli_given');
    soft_touch_vector = soft_touch_vector | map_events_to_neuron(cam1_file, 'A');
end

disp(['  Camera 1 — Soft_touch:     ', num2str(sum(soft_touch_vector)), ' neuron frames']);
disp(['  Camera 1 — Strong_touch:   ', num2str(sum(strong_touch_vector)), ' neuron frames']);
disp(['  Camera 1 — Mechanic_pain:  ', num2str(sum(mechanic_pain_vector)), ' neuron frames']);
disp(['  Camera 1 — Thermo_pain:    ', num2str(sum(thermo_pain_vector)), ' neuron frames']);

% ---- Process Camera 2 events ----
cam2_file = fullfile(output_folder, 'behavior_events_cam2.mat');
mouse_reaction_vector  = false(n_neuron_frames, 1);
reaction_offset_vector = false(n_neuron_frames, 1);

if opts.merge_both_cameras && isfile(cam2_file)
    mouse_reaction_vector  = map_events_to_neuron(cam2_file, 'Mouse_reaction');
    reaction_offset_vector = map_events_to_neuron(cam2_file, 'Reaction_offset');

    if sum(mouse_reaction_vector) == 0
        mouse_reaction_vector = map_events_to_neuron(cam2_file, 'A');
    end

    disp(['  Camera 2 — Mouse_reaction:  ', num2str(sum(mouse_reaction_vector)), ' neuron frames']);
    disp(['  Camera 2 — Reaction_offset: ', num2str(sum(reaction_offset_vector)), ' neuron frames']);
elseif opts.merge_both_cameras
    disp('  Camera 2 — behavior_events_cam2.mat not found (skipped)');
end

% Combined event vector (any event from either camera)
event_vector = soft_touch_vector | strong_touch_vector | ...
               mechanic_pain_vector | thermo_pain_vector | ...
               mouse_reaction_vector | reaction_offset_vector;

% ---- Collect original timestamps ----
event_timestamps_original = struct();
if isfile(cam1_file)
    tmp = load(cam1_file);
    event_timestamps_original.cam1_frames = tmp.event_frames;
    event_timestamps_original.cam1_labels = tmp.event_labels;
    if isfield(tmp, 'event_timestamps')
        event_timestamps_original.cam1_timestamps = tmp.event_timestamps;
    end
end
if isfile(cam2_file)
    tmp = load(cam2_file);
    event_timestamps_original.cam2_frames = tmp.event_frames;
    event_timestamps_original.cam2_labels = tmp.event_labels;
    if isfield(tmp, 'event_timestamps')
        event_timestamps_original.cam2_timestamps = tmp.event_timestamps;
    end
end

% ---- Build metadata ----
metadata = struct();
metadata.neuron_frame_rate_hz = neuron_frame_rate_hz;
metadata.behav_frame_rate_hz = ts.behav_frame_rate_hz;
metadata.n_neuron_frames = n_neuron_frames;
metadata.n_cells = n_cells;
metadata.n_soft_touch = sum(soft_touch_vector);
metadata.n_strong_touch = sum(strong_touch_vector);
metadata.n_mechanic_pain = sum(mechanic_pain_vector);
metadata.n_thermo_pain = sum(thermo_pain_vector);
metadata.n_mouse_reaction = sum(mouse_reaction_vector);
metadata.n_reaction_offset = sum(reaction_offset_vector);
metadata.n_any_event = sum(event_vector);
metadata.created = datetime('now');
metadata.columns = {'soft_touch_vector', 'strong_touch_vector', ...
                    'mechanic_pain_vector', 'thermo_pain_vector', ...
                    'mouse_reaction_vector', 'reaction_offset_vector', ...
                    'event_vector (any)'};

% ---- Save ----
out_file = fullfile(output_folder, 'final_neuron_behavior.mat');
save(out_file, 'deltaF_over_F', 'neuron_timestamps', 'neuron_seconds', ...
    'soft_touch_vector', 'strong_touch_vector', ...
    'mechanic_pain_vector', 'thermo_pain_vector', ...
    'mouse_reaction_vector', 'reaction_offset_vector', ...
    'event_vector', 'event_timestamps_original', ...
    'spatial_weights', 'metadata', '-v7.3');

disp(' ');
disp(['  Saved: ', out_file]);
disp(['    deltaF_over_F:          ', num2str(n_neuron_frames), ' x ', num2str(n_cells)]);
disp(['    soft_touch_vector:      ', num2str(sum(soft_touch_vector)), ' events']);
disp(['    strong_touch_vector:    ', num2str(sum(strong_touch_vector)), ' events']);
disp(['    mechanic_pain_vector:   ', num2str(sum(mechanic_pain_vector)), ' events']);
disp(['    thermo_pain_vector:     ', num2str(sum(thermo_pain_vector)), ' events']);
disp(['    mouse_reaction_vector:  ', num2str(sum(mouse_reaction_vector)), ' events']);
disp(['    reaction_offset_vector: ', num2str(sum(reaction_offset_vector)), ' events']);
disp(['    event_vector (any):     ', num2str(sum(event_vector)), ' events']);
disp('===== Step 5 complete =====');
end
