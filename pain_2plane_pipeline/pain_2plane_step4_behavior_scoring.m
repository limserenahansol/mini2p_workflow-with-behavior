function pain_2plane_step4_behavior_scoring(session_dir, camera_id, output_folder, timestamps_mat)
% Manual behavioral video scoring GUI.
%
% Camera 1 = Stimuli events (4 types)  |  Camera 2 = Mouse reaction events
%
% Loads all AVI files from one camera folder, concatenates them into a
% continuous timeline, and presents a figure-based video player for manual
% event annotation with per-camera event types.
%
% Controls:
%   Space        Play / Pause (starts at 5x speed)
%   Right arrow  Step forward 1 frame  (hold Shift: +10 frames)
%   Left arrow   Step backward 1 frame (hold Shift: -10 frames)
%
%   Camera 1 keys:
%     A   "Soft_touch"        (soft tactile stimulus)
%     S   "Strong_touch"      (strong tactile stimulus)
%     D   "Mechanic_pain"     (mechanical pain stimulus)
%     F   "Thermo_pain"       (thermal pain stimulus)
%
%   Camera 2 keys:
%     A   "Mouse_reaction"    (mouse behavioral response)
%     S   "Reaction_offset"   (end of response)
%
%   X            Delete last event
%   W            Save events to file
%   Q / Escape   Save and close
%   +/-          Speed up / slow down playback (1x,2x,5x,10x,30x)
%
% Usage:
%   pain_2plane_step4_behavior_scoring(session_dir, 1)          % cam 1: stimuli
%   pain_2plane_step4_behavior_scoring(session_dir, 2, out_dir) % cam 2: reaction
%   pain_2plane_step4_behavior_scoring(session_dir, 1, out_dir, 'timestamps.mat')
%
% Output:
%   <output_folder>/behavior_events_cam<N>.mat
%     .event_frames           - frame indices in concatenated video
%     .event_timestamps       - datetime values (if timestamps_mat provided)
%     .event_labels           - cell array of event type labels
%     .event_type_definitions - cell array of all possible labels for this camera
%     .avi_names              - list of AVI files used
%     .segment_boundaries     - cumulative frame count per AVI
%     .cam_id                 - which camera was scored

if nargin < 2 || isempty(camera_id), camera_id = 1; end
if nargin < 3 || isempty(output_folder)
    output_folder = fullfile(session_dir, 'output');
end
if nargin < 4, timestamps_mat = ''; end
if ~isfolder(output_folder), mkdir(output_folder); end

% ---- Camera-specific event definitions ----
% Each entry: {key, label, color}
if camera_id == 1
    cam_purpose = 'Stimuli camera';
    event_map = {
        'a', 'Soft_touch',     [0.2 0.8 0.2];   % green
        's', 'Strong_touch',   [1.0 0.6 0.0];   % orange
        'd', 'Mechanic_pain',  [1.0 0.2 0.2];   % red
        'f', 'Thermo_pain',    [0.8 0.2 1.0];   % purple
    };
else
    cam_purpose = 'Mouse reaction camera';
    event_map = {
        'a', 'Mouse_reaction',   [0.2 0.8 0.2];   % green
        's', 'Reaction_offset',  [0.2 0.6 1.0];   % blue
    };
end

% Build lookup
event_keys   = event_map(:,1);
event_labels_def = event_map(:,2);
event_colors = event_map(:,3);

disp('===== Step 4: Behavioral Video Scoring GUI =====');
disp(['  Camera ', num2str(camera_id), ': ', cam_purpose]);
for ei = 1:size(event_map, 1)
    disp(['    [', upper(event_map{ei,1}), '] = "', event_map{ei,2}, '"']);
end

% ---- Locate AVIs ----
if camera_id == 1
    avi_dir = fullfile(session_dir, 'MiceVideo1', 'MiceVideo');
else
    avi_dir = fullfile(session_dir, 'MiceVideo2', 'MiceVideo');
end

avi_files = dir(fullfile(avi_dir, '*.avi'));
if isempty(avi_files)
    error('No AVI files found in %s', avi_dir);
end
[~, sort_idx] = sort({avi_files.name});
avi_files = avi_files(sort_idx);
disp(['  AVI files: ', num2str(numel(avi_files))]);

% ---- Build video reader info ----
readers = cell(numel(avi_files), 1);
seg_nframes = zeros(numel(avi_files), 1);
for i = 1:numel(avi_files)
    fpath = fullfile(avi_files(i).folder, avi_files(i).name);
    readers{i} = VideoReader(fpath);
    seg_nframes(i) = readers{i}.NumFrames;
    disp(['    ', avi_files(i).name, ': ', num2str(seg_nframes(i)), ' frames']);
end
seg_boundaries = cumsum(seg_nframes);
total_frames = seg_boundaries(end);
disp(['  Total concatenated frames: ', num2str(total_frames)]);

% ---- Load timestamps (optional) ----
behav_ts = [];
if ~isempty(timestamps_mat) && isfile(timestamps_mat)
    ts_data = load(timestamps_mat);
    if camera_id == 1 && isfield(ts_data, 'behav1_timestamps')
        behav_ts = ts_data.behav1_timestamps;
    elseif camera_id == 2 && isfield(ts_data, 'behav2_timestamps')
        behav_ts = ts_data.behav2_timestamps;
    end
    if ~isempty(behav_ts) && numel(behav_ts) ~= total_frames
        warning('Timestamp count (%d) != total frames (%d). Timestamps may be misaligned.', ...
            numel(behav_ts), total_frames);
    end
end

% ---- Load existing events if re-scoring ----
existing_file = fullfile(output_folder, sprintf('behavior_events_cam%d.mat', camera_id));
loaded_events = false;

% ---- State ----
state = struct();
state.current_frame = 1;
state.playing = false;
state.playback_speed = 8.0;    % default 5x speed
state.event_frames = [];
state.event_labels = {};
state.total_frames = total_frames;
state.fps = readers{1}.FrameRate;
if isempty(state.fps) || state.fps <= 0, state.fps = 30; end

if isfile(existing_file)
    try
        prev = load(existing_file);
        if isfield(prev, 'event_frames') && isfield(prev, 'event_labels')
            state.event_frames = prev.event_frames(:)';
            state.event_labels = prev.event_labels(:)';
            loaded_events = true;
            disp(['  Loaded ', num2str(numel(state.event_frames)), ' existing events from previous session']);
        end
    catch
    end
end

% ---- Build GUI ----
key_hints = strjoin(cellfun(@(k,l) [upper(k),'=',l], event_keys, event_labels_def, 'Uni', false), '  ');
fig = figure('Name', sprintf('Cam %d: %s  |  %s', camera_id, cam_purpose, key_hints), ...
    'NumberTitle', 'off', 'Position', [50 50 1200 800], ...
    'CloseRequestFcn', @on_close, 'Color', [0.12 0.12 0.12]);

ax = axes('Parent', fig, 'Position', [0.02 0.18 0.96 0.78]);
axis(ax, 'off');

% Info panel (bottom left)
info_txt = uicontrol('Style', 'text', 'Parent', fig, ...
    'Units', 'normalized', 'Position', [0.02 0.01 0.50 0.12], ...
    'FontSize', 11, 'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.12 0.12 0.12], 'ForegroundColor', [0.9 0.9 0.9]);

% Event count panel (bottom right)
event_txt = uicontrol('Style', 'text', 'Parent', fig, ...
    'Units', 'normalized', 'Position', [0.53 0.01 0.45 0.12], ...
    'FontSize', 10, 'HorizontalAlignment', 'right', ...
    'BackgroundColor', [0.12 0.12 0.12], 'ForegroundColor', [1 1 0.4]);

% Slider
slider = uicontrol('Style', 'slider', 'Parent', fig, ...
    'Units', 'normalized', 'Position', [0.02 0.14 0.96 0.03], ...
    'Min', 1, 'Max', total_frames, 'Value', 1, ...
    'SliderStep', [1/total_frames, 100/total_frames], ...
    'Callback', @on_slider);

set(fig, 'KeyPressFcn', @on_key);

show_frame(1);
update_info();

% ---- Playback timer ----
% Fixed period (~30ms). Speed is achieved by skipping frames, not faster ticks.
play_timer = timer('ExecutionMode', 'fixedSpacing', ...
    'Period', 0.033, ...
    'TimerFcn', @on_timer_tick);

disp(' ');
disp('  Controls:');
disp('    Space      = Play/Pause (starts at 5x)');
disp('    Arrow L/R  = Step frame (hold Shift: 10 frames)');
for ei = 1:size(event_map, 1)
    disp(['    ', upper(event_map{ei,1}), '          = Mark "', event_map{ei,2}, '"']);
end
disp('    X          = Delete last event');
disp('    W          = Save    Q/Esc = Save & close');
disp('    +/-        = Speed: 1x,2x,5x,10x,15x,30x');
disp(' ');
uiwait(fig);

% ========== NESTED FUNCTIONS ==========

    function show_frame(idx)
        idx = max(1, min(total_frames, round(idx)));
        state.current_frame = idx;
        if ~isvalid(ax), return; end
        seg = find(idx <= seg_boundaries, 1, 'first');
        if seg == 1
            local_idx = idx;
        else
            local_idx = idx - seg_boundaries(seg-1);
        end
        readers{seg}.CurrentTime = (local_idx - 1) / readers{seg}.FrameRate;
        fr = readFrame(readers{seg});
        imshow(fr, 'Parent', ax);

        if ~isempty(state.event_frames)
            hold(ax, 'on');
            near_idx = find(abs(state.event_frames - idx) < 3);
            if ~isempty(near_idx)
                y_pos = 30;
                for ni = 1:numel(near_idx)
                    lbl = state.event_labels{near_idx(ni)};
                    clr = get_event_color(lbl);
                    text(ax, 20, y_pos, strrep(lbl, '_', ' '), 'Color', clr, ...
                        'FontSize', 16, 'FontWeight', 'bold');
                    y_pos = y_pos + 25;
                end
            end
            hold(ax, 'off');
        end
        if isvalid(slider), slider.Value = idx; end
    end

    function update_info()
        if ~isvalid(info_txt), return; end
        ts_str = '';
        if ~isempty(behav_ts) && state.current_frame <= numel(behav_ts)
            ts_str = ['  |  ', char(behav_ts(state.current_frame), 'HH:mm:ss.SSS')];
        end
        elapsed = (state.current_frame - 1) / state.fps;
        play_str = 'PAUSED';
        if state.playing, play_str = sprintf('PLAYING x%.1f', state.playback_speed); end
        info_txt.String = sprintf('Frame %d / %d  |  %.1fs  |  %s%s', ...
            state.current_frame, total_frames, elapsed, play_str, ts_str);

        count_parts = {};
        for ei = 1:numel(event_labels_def)
            n = sum(strcmp(state.event_labels, event_labels_def{ei}));
            count_parts{end+1} = sprintf('%s:%d', event_labels_def{ei}, n); %#ok<AGROW>
        end
        if isvalid(event_txt)
            event_txt.String = [strjoin(count_parts, '  |  '), ...
                sprintf('  |  Total: %d', numel(state.event_frames))];
        end
    end

    function on_key(~, evt)
        has_shift = any(strcmp(evt.Modifier, 'shift'));
        switch evt.Key
            case 'space'
                toggle_play();
            case 'rightarrow'
                stop_play();
                step = 1;
                if has_shift, step = 10; end
                show_frame(state.current_frame + step);
                update_info();
            case 'leftarrow'
                stop_play();
                step = 1;
                if has_shift, step = 10; end
                show_frame(state.current_frame - step);
                update_info();
            case {'a','s','d','f'}
                ki = find(strcmp(event_keys, evt.Key));
                if ~isempty(ki)
                    add_event(event_labels_def{ki});
                end
            case 'x'
                delete_last_event();
            case 'w'
                save_events();
            case {'q', 'escape'}
                save_events();
                on_close();
            case 'equal'
                state.playback_speed = min(30, state.playback_speed * 2);
                update_timer_period();
                update_info();
            case 'hyphen'
                state.playback_speed = max(1, state.playback_speed / 2);
                update_timer_period();
                update_info();
        end
    end

    function toggle_play()
        if state.playing
            stop_play();
        else
            state.playing = true;
            update_timer_period();
            start(play_timer);
        end
        update_info();
    end

    function stop_play()
        state.playing = false;
        if strcmp(play_timer.Running, 'on')
            stop(play_timer);
        end
    end

    function update_timer_period()
        % Timer period is fixed; speed changes take effect via frame skipping
        % in on_timer_tick. Nothing to update here.
    end

    function on_timer_tick(~, ~)
        if ~isvalid(fig), return; end
        if state.current_frame >= total_frames
            stop_play();
            update_info();
            return;
        end
        frame_step = max(1, round(state.playback_speed));
        show_frame(state.current_frame + frame_step);
        update_info();
    end

    function on_slider(src, ~)
        stop_play();
        show_frame(round(src.Value));
        update_info();
    end

    function add_event(label)
        fr = state.current_frame;
        % Allow same frame only if different event type
        already = find(state.event_frames == fr);
        for ai = 1:numel(already)
            if strcmp(state.event_labels{already(ai)}, label)
                disp(['  (duplicate skipped: ', label, ' already at frame ', num2str(fr), ')']);
                return;
            end
        end
        state.event_frames(end+1) = fr;
        state.event_labels{end+1} = label;
        disp(['  + ', label, ' at frame ', num2str(fr)]);
        update_info();
        show_frame(fr);
    end

    function delete_last_event()
        if ~isempty(state.event_frames)
            removed_fr = state.event_frames(end);
            removed_lbl = state.event_labels{end};
            state.event_frames(end) = [];
            state.event_labels(end) = [];
            disp(['  - Removed ', removed_lbl, ' at frame ', num2str(removed_fr)]);
        end
        update_info();
        show_frame(state.current_frame);
    end

    function save_events()
        [event_frames_sorted, si] = sort(state.event_frames(:));
        event_frames = event_frames_sorted; %#ok<NASGU>
        event_labels = state.event_labels(si); %#ok<NASGU>
        event_labels = event_labels(:); %#ok<NASGU>

        event_timestamps = []; %#ok<NASGU>
        if ~isempty(behav_ts)
            valid = event_frames_sorted(event_frames_sorted <= numel(behav_ts));
            event_timestamps = behav_ts(valid); %#ok<NASGU>
        end

        event_type_definitions = event_labels_def(:); %#ok<NASGU>
        avi_names = {avi_files.name}'; %#ok<NASGU>
        segment_boundaries = seg_boundaries; %#ok<NASGU>
        cam_id = camera_id; %#ok<NASGU>
        cam_purpose_str = cam_purpose; %#ok<NASGU>

        out_file = fullfile(output_folder, sprintf('behavior_events_cam%d.mat', camera_id));
        save(out_file, 'event_frames', 'event_timestamps', 'event_labels', ...
            'event_type_definitions', 'avi_names', 'segment_boundaries', ...
            'cam_id', 'cam_purpose_str', '-v7.3');
        disp(['  Saved: ', out_file]);

        for ei = 1:numel(event_labels_def)
            n = sum(strcmp(state.event_labels, event_labels_def{ei}));
            disp(['    ', event_labels_def{ei}, ': ', num2str(n)]);
        end
    end

    function clr = get_event_color(lbl)
        ki = find(strcmp(event_labels_def, lbl), 1);
        if ~isempty(ki)
            clr = event_colors{ki};
        else
            clr = [1 1 1];
        end
    end

    function on_close(~, ~)
        stop_play();
        try stop(play_timer); end %#ok<TRYNC>
        try delete(play_timer); end %#ok<TRYNC>
        for ri = 1:numel(readers)
            try delete(readers{ri}); end %#ok<TRYNC>
        end
        delete(fig);
    end

end
