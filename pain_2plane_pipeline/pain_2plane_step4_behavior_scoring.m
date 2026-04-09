function pain_2plane_step4_behavior_scoring(session_dir, camera_id, output_folder, timestamps_mat)
% Manual behavioral video scoring GUI.
%
% Loads all AVI files from one camera folder, concatenates them into a
% continuous timeline, and presents a figure-based video player for manual
% event annotation.
%
% Controls:
%   Space       Play / Pause
%   Right arrow  Step forward 1 frame
%   Left arrow   Step backward 1 frame
%   A            Mark stimulus event at current frame
%   D            Delete last event
%   S            Save events to file
%   Q / Escape   Save and close
%   +/-          Speed up / slow down playback
%
% Usage:
%   pain_2plane_step4_behavior_scoring(session_dir, 1)          % camera 1
%   pain_2plane_step4_behavior_scoring(session_dir, 2, out_dir) % camera 2
%   pain_2plane_step4_behavior_scoring(session_dir, 1, out_dir, 'timestamps.mat')
%
% Output:
%   <output_folder>/behavior_events_cam<N>.mat
%     .event_frames      - frame indices in concatenated video
%     .event_timestamps  - datetime values (if timestamps_mat provided)
%     .event_labels      - cell array of event types (all 'A')
%     .avi_files          - list of AVI files used
%     .segment_boundaries - cumulative frame count per AVI
%     .camera_id          - which camera was scored

if nargin < 2 || isempty(camera_id), camera_id = 1; end
if nargin < 3 || isempty(output_folder)
    output_folder = fullfile(session_dir, 'output');
end
if nargin < 4, timestamps_mat = ''; end
if ~isfolder(output_folder), mkdir(output_folder); end

disp('===== Step 4: Behavioral Video Scoring GUI =====');

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
% Sort by name (chronological)
[~, sort_idx] = sort({avi_files.name});
avi_files = avi_files(sort_idx);
disp(['  Camera ', num2str(camera_id), ': ', num2str(numel(avi_files)), ' AVI files']);

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

% ---- State ----
state = struct();
state.current_frame = 1;
state.playing = false;
state.playback_speed = 1.0;
state.event_frames = [];
state.event_labels = {};
state.total_frames = total_frames;
state.fps = readers{1}.FrameRate;
if isempty(state.fps) || state.fps <= 0, state.fps = 30; end

% ---- Build GUI ----
fig = figure('Name', sprintf('Behavior Scoring - Camera %d', camera_id), ...
    'NumberTitle', 'off', 'Position', [50 50 1100 750], ...
    'CloseRequestFcn', @on_close, 'Color', [0.15 0.15 0.15]);

ax = axes('Parent', fig, 'Position', [0.02 0.15 0.96 0.80]);
axis(ax, 'off');

% Info text
info_txt = uicontrol('Style', 'text', 'Parent', fig, ...
    'Units', 'normalized', 'Position', [0.02 0.01 0.55 0.10], ...
    'FontSize', 11, 'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.15 0.15 0.15], 'ForegroundColor', [0.9 0.9 0.9]);

% Event count text
event_txt = uicontrol('Style', 'text', 'Parent', fig, ...
    'Units', 'normalized', 'Position', [0.58 0.01 0.40 0.10], ...
    'FontSize', 11, 'HorizontalAlignment', 'right', ...
    'BackgroundColor', [0.15 0.15 0.15], 'ForegroundColor', [1 1 0]);

% Slider
slider = uicontrol('Style', 'slider', 'Parent', fig, ...
    'Units', 'normalized', 'Position', [0.02 0.11 0.96 0.03], ...
    'Min', 1, 'Max', total_frames, 'Value', 1, ...
    'SliderStep', [1/total_frames, 100/total_frames], ...
    'Callback', @on_slider);

% Key handler
set(fig, 'KeyPressFcn', @on_key);

% Display first frame
show_frame(1);
update_info();

% ---- Playback timer ----
play_timer = timer('ExecutionMode', 'fixedSpacing', ...
    'Period', max(0.01, round(1/state.fps/state.playback_speed, 3)), ...
    'TimerFcn', @on_timer_tick);

disp('  GUI ready. Controls: Space=play/pause, A=mark event, D=undo, S=save, Q=quit');
uiwait(fig);

% ---- Nested functions ----

    function show_frame(idx)
        idx = max(1, min(total_frames, round(idx)));
        state.current_frame = idx;
        % Find which segment
        seg = find(idx <= seg_boundaries, 1, 'first');
        if seg == 1
            local_idx = idx;
        else
            local_idx = idx - seg_boundaries(seg-1);
        end
        readers{seg}.CurrentTime = (local_idx - 1) / readers{seg}.FrameRate;
        fr = readFrame(readers{seg});
        imshow(fr, 'Parent', ax);
        % Draw event markers on frame
        if ~isempty(state.event_frames)
            hold(ax, 'on');
            near = abs(state.event_frames - idx) < 3;
            if any(near)
                text(ax, 20, 30, 'EVENT', 'Color', 'r', 'FontSize', 18, 'FontWeight', 'bold');
            end
            hold(ax, 'off');
        end
        slider.Value = idx;
    end

    function update_info()
        ts_str = '';
        if ~isempty(behav_ts) && state.current_frame <= numel(behav_ts)
            ts_str = ['  |  ', char(behav_ts(state.current_frame), 'HH:mm:ss.SSS')];
        end
        elapsed = (state.current_frame - 1) / state.fps;
        play_str = 'PAUSED';
        if state.playing, play_str = sprintf('PLAYING x%.1f', state.playback_speed); end
        info_txt.String = sprintf('Frame %d / %d  |  %.1f s  |  %s%s', ...
            state.current_frame, total_frames, elapsed, play_str, ts_str);
        event_txt.String = sprintf('Events: %d  |  [A]=mark  [D]=undo  [S]=save  [Q]=quit', ...
            numel(state.event_frames));
    end

    function on_key(~, evt)
        switch evt.Key
            case 'space'
                toggle_play();
            case 'rightarrow'
                stop_play();
                show_frame(state.current_frame + 1);
                update_info();
            case 'leftarrow'
                stop_play();
                show_frame(state.current_frame - 1);
                update_info();
            case 'a'
                add_event();
            case 'd'
                delete_last_event();
            case 's'
                save_events();
            case {'q', 'escape'}
                save_events();
                on_close();
            case 'equal'
                state.playback_speed = min(8, state.playback_speed * 2);
                update_timer_period();
                update_info();
            case 'hyphen'
                state.playback_speed = max(0.125, state.playback_speed / 2);
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
        if strcmp(play_timer.Running, 'on')
            stop(play_timer);
        end
        play_timer.Period = max(0.01, round(1/state.fps/state.playback_speed, 3));
        if state.playing
            start(play_timer);
        end
    end

    function on_timer_tick(~, ~)
        if state.current_frame >= total_frames
            stop_play();
            update_info();
            return;
        end
        show_frame(state.current_frame + 1);
        update_info();
    end

    function on_slider(src, ~)
        stop_play();
        show_frame(round(src.Value));
        update_info();
    end

    function add_event()
        fr = state.current_frame;
        if ~ismember(fr, state.event_frames)
            state.event_frames(end+1) = fr;
            state.event_labels{end+1} = 'A';
            disp(['  + Event A at frame ', num2str(fr)]);
        end
        update_info();
        show_frame(fr);
    end

    function delete_last_event()
        if ~isempty(state.event_frames)
            removed = state.event_frames(end);
            state.event_frames(end) = [];
            state.event_labels(end) = [];
            disp(['  - Removed event at frame ', num2str(removed)]);
        end
        update_info();
        show_frame(state.current_frame);
    end

    function save_events()
        event_frames = sort(state.event_frames(:));
        event_labels = state.event_labels(:);
        if numel(event_labels) ~= numel(event_frames)
            [~, si] = sort(state.event_frames);
            event_labels = state.event_labels(si);
        end

        event_timestamps = [];
        if ~isempty(behav_ts)
            valid = event_frames(event_frames <= numel(behav_ts));
            event_timestamps = behav_ts(valid);
        end

        avi_names = {avi_files.name}'; %#ok<NASGU>
        segment_boundaries = seg_boundaries; %#ok<NASGU>
        cam_id = camera_id; %#ok<NASGU>

        out_file = fullfile(output_folder, sprintf('behavior_events_cam%d.mat', camera_id));
        save(out_file, 'event_frames', 'event_timestamps', 'event_labels', ...
            'avi_names', 'segment_boundaries', 'cam_id', '-v7.3');
        disp(['  Saved: ', out_file, ' (', num2str(numel(event_frames)), ' events)']);
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
