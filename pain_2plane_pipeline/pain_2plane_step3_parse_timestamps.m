function pain_2plane_step3_parse_timestamps(session_dir, output_folder)
% Parse TDMS timestamp files for neuron and behavioral camera data.
%
% Reads:
%   CellVideo1/CellVideo_CHA_Info.tdms   -> per-frame neuron timestamps (18000)
%   MiceVideo1/MiceVideo...-reference.tdms -> camera 1 timestamps (~58640 @ 30 Hz)
%   MiceVideo2/MiceVideo...-reference.tdms -> camera 2 timestamps (~58637 @ 30 Hz)
%   SyncInformation/SignalSync_*.tdms      -> sync pulse timestamps (optional)
%
% After max-projection pairing (Step 1), we keep the timestamp of each
% pair's first frame (odd-indexed raw frames), giving 9000 neuron timestamps.
%
% Output:
%   <output_folder>/timestamps.mat
%     .neuron_timestamps_all    (18000x1 datetime)   raw per-frame
%     .neuron_timestamps        (9000x1  datetime)   after pair reduction
%     .neuron_seconds           (9000x1  double)     seconds since first frame
%     .behav1_timestamps        (Nx1     datetime)   camera 1
%     .behav1_seconds           (Nx1     double)     seconds since first cam1 frame
%     .behav2_timestamps        (Mx1     datetime)   camera 2
%     .behav2_seconds           (Mx1     double)
%     .sync_timestamps          (cell)               sync channel timestamps
%     .neuron_frame_rate_hz     double               effective rate after pair merge
%     .behav_frame_rate_hz      double               behavioral camera rate
%
% Usage:
%   pain_2plane_step3_parse_timestamps(session_dir, output_folder)

if nargin < 2 || isempty(output_folder)
    output_folder = fullfile(session_dir, 'output');
end
if ~isfolder(output_folder), mkdir(output_folder); end

disp('===== Step 3: Parse TDMS Timestamps =====');

% ---- Neuron timestamps ----
neuron_tdms = find_tdms(fullfile(session_dir, 'CellVideo1'), '*Info.tdms');
disp(['  Neuron TDMS: ', neuron_tdms]);
neuron_data = tdmsread(neuron_tdms);

% Find the group containing frame timestamps (group "CHA" has channel "Time")
neuron_timestamps_all = [];
for gi = 1:numel(neuron_data)
    tbl = neuron_data{gi};
    if any(strcmpi(tbl.Properties.VariableNames, 'Time'))
        time_raw = tbl.Time;
        if iscell(time_raw)
            time_raw = string(time_raw);
        end
        neuron_timestamps_all = datetime(time_raw, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
        break;
    end
end
if isempty(neuron_timestamps_all)
    error('Could not find "Time" channel in neuron TDMS.');
end
disp(['  Neuron frames (raw): ', num2str(numel(neuron_timestamps_all))]);

% After max-projection: take timestamp of each pair's first frame (odd frames)
odd_idx = 1:2:numel(neuron_timestamps_all);
neuron_timestamps = neuron_timestamps_all(odd_idx);
neuron_seconds = seconds(neuron_timestamps - neuron_timestamps(1));
neuron_frame_rate_hz = 1 / median(diff(neuron_seconds));
disp(['  Neuron frames (paired): ', num2str(numel(neuron_timestamps))]);
disp(['  Effective neuron rate:   ', num2str(neuron_frame_rate_hz, '%.2f'), ' Hz']);

% ---- Behavioral camera 1 ----
behav1_tdms = find_tdms(fullfile(session_dir, 'MiceVideo1'), '*reference.tdms');
disp(['  Behavior cam1 TDMS: ', behav1_tdms]);
behav1_data = tdmsread(behav1_tdms);
behav1_timestamps = extract_ref_time(behav1_data);
behav1_seconds = seconds(behav1_timestamps - behav1_timestamps(1));
disp(['  Cam1 frames: ', num2str(numel(behav1_timestamps))]);

% ---- Behavioral camera 2 ----
behav2_tdms = find_tdms(fullfile(session_dir, 'MiceVideo2'), '*reference.tdms');
disp(['  Behavior cam2 TDMS: ', behav2_tdms]);
behav2_data = tdmsread(behav2_tdms);
behav2_timestamps = extract_ref_time(behav2_data);
behav2_seconds = seconds(behav2_timestamps - behav2_timestamps(1));
disp(['  Cam2 frames: ', num2str(numel(behav2_timestamps))]);

behav_frame_rate_hz = 1 / median(diff(behav1_seconds));
disp(['  Behavioral camera rate:  ', num2str(behav_frame_rate_hz, '%.2f'), ' Hz']);

% ---- Sync signals (optional) ----
sync_dir = fullfile(session_dir, 'SyncInformation');
sync_timestamps = {};
if isfolder(sync_dir)
    sync_files = dir(fullfile(sync_dir, 'SignalSync_*.tdms'));
    sync_files = sync_files(~contains({sync_files.name}, '_index'));
    for si = 1:numel(sync_files)
        sf = fullfile(sync_files(si).folder, sync_files(si).name);
        try
            sd = tdmsread(sf);
            for gi = 1:numel(sd)
                tbl = sd{gi};
                if any(strcmpi(tbl.Properties.VariableNames, 'Time'))
                    tr = tbl.Time;
                    if iscell(tr), tr = string(tr); end
                    ts = datetime(tr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
                    sync_timestamps{end+1} = ts; %#ok<AGROW>
                    disp(['  Sync file ', sync_files(si).name, ': ', num2str(numel(ts)), ' events']);
                    break;
                end
            end
        catch ME
            disp(['  Warning: could not read ', sync_files(si).name, ': ', ME.message]);
        end
    end
end

% ---- Save ----
out_path = fullfile(output_folder, 'timestamps.mat');
save(out_path, 'neuron_timestamps_all', 'neuron_timestamps', 'neuron_seconds', ...
    'behav1_timestamps', 'behav1_seconds', 'behav2_timestamps', 'behav2_seconds', ...
    'sync_timestamps', 'neuron_frame_rate_hz', 'behav_frame_rate_hz', '-v7.3');
disp(['  Saved: ', out_path]);
disp('===== Step 3 complete =====');
end

% ===== LOCAL HELPERS =====

function fpath = find_tdms(folder, pattern)
    files = dir(fullfile(folder, pattern));
    files = files(~contains({files.name}, '_index'));
    if isempty(files)
        error('No TDMS matching "%s" in %s', pattern, folder);
    end
    fpath = fullfile(files(1).folder, files(1).name);
end

function ts = extract_ref_time(tdms_data)
    ts = [];
    for gi = 1:numel(tdms_data)
        tbl = tdms_data{gi};
        colnames = tbl.Properties.VariableNames;
        % Look for "Ref Time" or "RefTime" or "Ref_Time"
        ref_col = colnames(contains(colnames, 'Ref', 'IgnoreCase', true) & ...
                           contains(colnames, 'Time', 'IgnoreCase', true));
        if ~isempty(ref_col)
            tr = tbl.(ref_col{1});
            if iscell(tr), tr = string(tr); end
            ts = datetime(tr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
            return;
        end
        % Fallback: look for "Time" column
        if any(strcmpi(colnames, 'Time'))
            tr = tbl.Time;
            if iscell(tr), tr = string(tr); end
            ts = datetime(tr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
            return;
        end
    end
    error('No timestamp channel found in behavioral TDMS.');
end
