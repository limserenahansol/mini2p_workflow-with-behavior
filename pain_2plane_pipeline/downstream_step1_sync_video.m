function downstream_step1_sync_video(session_dir, output_folder, opts)
% Generate a synchronized 3-panel video clip:
%   Top-left:  Camera 1 (stimuli) with event markers
%   Top-right: Camera 2 (mouse reaction) with event markers
%   Bottom:    Neuron dF/F heatmap (all cells) scrolling in time
%
% The clip starts at a user-defined neuron frame and runs for a set
% duration, all played at a configurable speed multiplier.
%
% Required inputs:
%   session_dir   - root session folder (contains CellVideo1, MiceVideo1, MiceVideo2)
%   output_folder - path to the output folder with final_neuron_behavior.mat
%
% Optional fields in opts:
%   .start_neuron_frame  - first neuron frame to show (default 4400)
%   .clip_duration_sec   - real-time duration of clip in seconds (default 180 = 3 min)
%   .speed_multiplier    - playback speed (default 5)
%   .output_fps          - output video frame rate (default 30)
%   .output_filename     - name of output AVI (default 'sync_3view_clip.avi')
%   .cam1_subfolder      - camera 1 folder name (default 'MiceVideo1')
%   .cam2_subfolder      - camera 2 folder name (default 'MiceVideo2')
%   .neuron_frame_rate   - Hz (default 4.605)
%   .behav_frame_rate    - Hz (default 30)
%   .video_width         - output width in pixels (default 1280)
%   .video_height        - output height in pixels (default 720)

if nargin < 3, opts = struct(); end

def = @(f,v) tern(isfield(opts,f), opts.(f), v);
start_nf       = def('start_neuron_frame', 4400);
clip_dur       = def('clip_duration_sec', 180);
speed_mult     = def('speed_multiplier', 5);
out_fps        = def('output_fps', 30);
out_name       = def('output_filename', 'sync_3view_clip.avi');
cam1_sub       = def('cam1_subfolder', 'MiceVideo1');
cam2_sub       = def('cam2_subfolder', 'MiceVideo2');
nfr            = def('neuron_frame_rate', 4.605);
bfr            = def('behav_frame_rate', 30);
vid_w          = def('video_width', 1280);
vid_h          = def('video_height', 720);

    function v = tern(c,a,b)
        if c, v = a; else, v = b; end
    end

disp('===== Downstream Step 1: Synchronized 3-View Video =====');

% ---- Load merged data ----
merged = load(fullfile(output_folder, 'final_neuron_behavior.mat'));
dff = merged.deltaF_over_F;
n_neuron_total = size(dff, 1);
n_cells = size(dff, 2);

event_names = {'Soft_touch','Strong_touch','Mechanic_pain','Thermo_pain'};
event_colors_rgb = [0 200 0; 255 150 0; 255 50 50; 200 50 255];
event_vectors_cam1 = [merged.soft_touch_vector, merged.strong_touch_vector, ...
                      merged.mechanic_pain_vector, merged.thermo_pain_vector];

reaction_names = {'Mouse_reaction','Reaction_offset'};
reaction_colors_rgb = [0 200 0; 50 150 255];
event_vectors_cam2 = [merged.mouse_reaction_vector, merged.reaction_offset_vector];

% Neuron frame range
end_nf = min(n_neuron_total, start_nf + round(clip_dur * nfr) - 1);
nf_range = start_nf:end_nf;
n_clip_nf = numel(nf_range);
disp(['  Neuron frames: ', num2str(start_nf), ' to ', num2str(end_nf), ...
      ' (', num2str(n_clip_nf), ' frames, ', sprintf('%.1f', n_clip_nf/nfr), 's)']);

% Behavioral frame range (corresponding)
ratio = bfr / nfr;
start_bf = max(1, round((start_nf - 1) * ratio) + 1);
end_bf   = round((end_nf) * ratio);

% ---- Load camera videos ----
% AVIs are in <session>/MiceVideo1/MiceVideo/*.avi (nested subfolder)
cam1_dir = fullfile(session_dir, cam1_sub, 'MiceVideo');
cam2_dir = fullfile(session_dir, cam2_sub, 'MiceVideo');

% Fall back to direct subfolder if nested 'MiceVideo' doesn't exist
if ~isfolder(cam1_dir)
    cam1_dir = fullfile(session_dir, cam1_sub);
end
if ~isfolder(cam2_dir)
    cam2_dir = fullfile(session_dir, cam2_sub);
end

[cam1_readers, cam1_seg] = load_camera_avis(cam1_dir);
[cam2_readers, cam2_seg] = load_camera_avis(cam2_dir);

disp(['  Camera 1: ', num2str(numel(cam1_readers)), ' AVI segments, ', ...
      num2str(cam1_seg(end)), ' total frames']);
disp(['  Camera 2: ', num2str(numel(cam2_readers)), ' AVI segments, ', ...
      num2str(cam2_seg(end)), ' total frames']);

% ---- Prepare dF/F heatmap data for the clip ----
dff_clip = dff(nf_range, :)';
dff_clip_norm = dff_clip;
for ci = 1:n_cells
    mn = prctile(dff_clip(ci,:), 5);
    mx = prctile(dff_clip(ci,:), 99);
    if mx > mn
        dff_clip_norm(ci,:) = (dff_clip(ci,:) - mn) / (mx - mn);
    end
end
dff_clip_norm = max(0, min(1, dff_clip_norm));

% ---- Create output video ----
out_path = fullfile(output_folder, out_name);
vw = VideoWriter(out_path, 'Motion JPEG AVI');
vw.FrameRate = out_fps;
vw.Quality = 90;
open(vw);

% Layout: top half = two camera panels, bottom half = dF/F heatmap
cam_h = round(vid_h * 0.55);
cam_w = round(vid_w / 2);
heat_h = vid_h - cam_h;

real_time_per_out_frame = speed_mult / out_fps;
total_out_frames = round(clip_dur / speed_mult * out_fps);
disp(['  Output: ', num2str(total_out_frames), ' frames at ', ...
      num2str(out_fps), ' fps (', sprintf('%.1f', total_out_frames/out_fps), ...
      's video for ', sprintf('%.0f', clip_dur), 's real time at ', ...
      num2str(speed_mult), 'x)']);

disp('  Rendering...');
for fi = 1:total_out_frames
    real_elapsed = (fi - 1) * real_time_per_out_frame;

    % Current neuron frame
    cur_nf = start_nf + round(real_elapsed * nfr);
    cur_nf = min(cur_nf, end_nf);
    cur_nf_idx = cur_nf - start_nf + 1;

    % Current behavioral frame
    cur_bf = start_bf + round(real_elapsed * bfr);
    cur_bf = min(cur_bf, end_bf);

    canvas = zeros(vid_h, vid_w, 3, 'uint8');

    % --- Camera 1 panel ---
    fr1 = read_behav_frame(cam1_readers, cam1_seg, cur_bf);
    fr1 = imresize(fr1, [cam_h, cam_w]);
    % Overlay event markers
    active_cam1 = find_active_events(event_vectors_cam1, cur_nf, n_neuron_total);
    fr1 = overlay_event_text(fr1, active_cam1, event_names, event_colors_rgb, 'CAM1: Stimuli');
    canvas(1:cam_h, 1:cam_w, :) = fr1;

    % --- Camera 2 panel ---
    fr2 = read_behav_frame(cam2_readers, cam2_seg, cur_bf);
    fr2 = imresize(fr2, [cam_h, cam_w]);
    active_cam2 = find_active_events(event_vectors_cam2, cur_nf, n_neuron_total);
    fr2 = overlay_event_text(fr2, active_cam2, reaction_names, reaction_colors_rgb, 'CAM2: Reaction');
    canvas(1:cam_h, cam_w+1:vid_w, :) = fr2;

    % --- dF/F heatmap panel ---
    heat_img = render_heatmap(dff_clip_norm, cur_nf_idx, n_clip_nf, vid_w, heat_h, ...
        nf_range, nfr, event_vectors_cam1, start_nf, event_colors_rgb);
    canvas(cam_h+1:vid_h, 1:vid_w, :) = heat_img;

    writeVideo(vw, canvas);

    if mod(fi, 100) == 0
        fprintf('    %d / %d (%.0f%%)\n', fi, total_out_frames, fi/total_out_frames*100);
    end
end

close(vw);
disp(['  Saved: ', out_path]);
disp('===== Downstream Step 1 complete =====');

end

% =========================================================================
%  HELPER FUNCTIONS
% =========================================================================

function [readers, seg_bounds] = load_camera_avis(cam_dir)
    avis = dir(fullfile(cam_dir, '*.avi'));
    if isempty(avis)
        error('No AVI files found in %s', cam_dir);
    end
    [~, order] = sort({avis.name});
    avis = avis(order);
    readers = cell(numel(avis), 1);
    seg_bounds = zeros(numel(avis), 1);
    cumf = 0;
    for i = 1:numel(avis)
        readers{i} = VideoReader(fullfile(cam_dir, avis(i).name));
        cumf = cumf + readers{i}.NumFrames;
        seg_bounds(i) = cumf;
    end
end

function fr = read_behav_frame(readers, seg_bounds, global_idx)
    global_idx = max(1, min(global_idx, seg_bounds(end)));
    seg = find(global_idx <= seg_bounds, 1, 'first');
    if seg == 1
        local = global_idx;
    else
        local = global_idx - seg_bounds(seg - 1);
    end
    local = max(1, min(local, readers{seg}.NumFrames));
    readers{seg}.CurrentTime = (local - 1) / readers{seg}.FrameRate;
    fr = readFrame(readers{seg});
end

function active = find_active_events(event_mat, cur_nf, n_total)
    cur_nf = max(1, min(cur_nf, n_total));
    window = max(1, cur_nf-1):min(n_total, cur_nf+1);
    active = false(1, size(event_mat, 2));
    for k = 1:size(event_mat, 2)
        active(k) = any(event_mat(window, k));
    end
end

function fr = overlay_event_text(fr, active, names, colors, title_str)
    fr = insertText(fr, [5, 5], title_str, 'FontSize', 14, ...
        'TextColor', 'white', 'BoxOpacity', 0.5, 'BoxColor', 'black');
    y = 30;
    for k = 1:numel(names)
        if active(k)
            fr = insertText(fr, [5, y], ['>> ', strrep(names{k},'_',' ')], ...
                'FontSize', 16, 'TextColor', colors(k,:), ...
                'BoxOpacity', 0.6, 'BoxColor', 'black');
            y = y + 28;
        end
    end
end

function img = render_heatmap(dff_norm, cur_idx, n_frames, w, h, nf_range, nfr, ...
        event_vecs, start_nf, event_colors)
    n_cells = size(dff_norm, 1);

    % Heatmap (cells x time) — show full clip with playhead
    heatmap_raw = ind2rgb(round(dff_norm * 255) + 1, parula(256));
    heatmap_img = imresize(heatmap_raw, [max(h-30, 1), w], 'nearest');
    heatmap_uint = im2uint8(heatmap_img);

    % Draw playhead line
    px = max(1, min(w, round(cur_idx / n_frames * w)));
    heatmap_uint(:, max(1,px-1):min(w,px+1), 1) = 255;
    heatmap_uint(:, max(1,px-1):min(w,px+1), 2) = 255;
    heatmap_uint(:, max(1,px-1):min(w,px+1), 3) = 255;

    % Draw event tick marks at top of heatmap
    for ev = 1:size(event_vecs, 2)
        ev_frames = find(event_vecs(start_nf:start_nf+n_frames-1, ev));
        for ei = 1:numel(ev_frames)
            ex = max(1, min(w, round(ev_frames(ei) / n_frames * w)));
            for ch = 1:3
                heatmap_uint(1:min(6,size(heatmap_uint,1)), max(1,ex-1):min(w,ex+1), ch) = event_colors(min(ev,size(event_colors,1)), ch);
            end
        end
    end

    % Time axis bar at bottom
    time_bar = zeros(30, w, 3, 'uint8');
    time_bar(:,:,:) = 20;
    cur_sec = (nf_range(max(1,cur_idx)) - nf_range(1)) / nfr;
    total_sec = numel(nf_range) / nfr;
    time_bar = insertText(time_bar, [5, 2], ...
        sprintf('dF/F  |  t=%.1fs / %.1fs  |  Cells: %d  |  Frame %d', ...
        cur_sec, total_sec, n_cells, nf_range(max(1,cur_idx))), ...
        'FontSize', 12, 'TextColor', 'white', 'BoxOpacity', 0);

    img = [heatmap_uint; time_bar(1:30, 1:w, :)];
    img = img(1:h, 1:w, :);
end
