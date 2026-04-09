function pain_2plane_step1_merge_planes(session_dir, output_folder)
% Load interleaved 2-plane TIF stacks from CellVideo1/CellVideo/,
% max-project each consecutive pair (planeA, planeB) into a single frame,
% and save the combined movie as combined_maxproj.tif.
%
% Usage:
%   pain_2plane_step1_merge_planes(session_dir, output_folder)

if nargin < 2 || isempty(output_folder)
    output_folder = fullfile(session_dir, 'output');
end
if ~isfolder(output_folder), mkdir(output_folder); end

tif_dir = fullfile(session_dir, 'CellVideo1', 'CellVideo');
if ~isfolder(tif_dir)
    error('CellVideo folder not found: %s', tif_dir);
end

tif_files = dir(fullfile(tif_dir, '*.tif'));
if isempty(tif_files)
    tif_files = dir(fullfile(tif_dir, '*.tiff'));
end
if isempty(tif_files)
    error('No TIF files found in %s', tif_dir);
end

% Sort by natural numeric order (CellVideo 1.tif, CellVideo 2.tif, ...)
names = {tif_files.name};
nums = zeros(1, numel(names));
for i = 1:numel(names)
    tok = regexp(names{i}, '(\d+)', 'tokens');
    if ~isempty(tok)
        nums(i) = str2double(tok{end}{1});
    end
end
[~, sort_idx] = sort(nums);
tif_files = tif_files(sort_idx);

disp('===== Step 1: Load and Merge 2-Plane TIFs =====');
disp(['  TIF folder: ', tif_dir]);
disp(['  Files found: ', num2str(numel(tif_files))]);

% First pass: count total frames
total_raw = 0;
for f = 1:numel(tif_files)
    fpath = fullfile(tif_files(f).folder, tif_files(f).name);
    info = imfinfo(fpath);
    total_raw = total_raw + numel(info);
end
disp(['  Total raw frames: ', num2str(total_raw)]);

if mod(total_raw, 2) ~= 0
    warning('Odd total frame count (%d). Last frame will be dropped.', total_raw);
end
n_pairs = floor(total_raw / 2);
disp(['  Max-projected frames: ', num2str(n_pairs)]);

% Read first frame to get dimensions
info1 = imfinfo(fullfile(tif_files(1).folder, tif_files(1).name));
h = info1(1).Height;
w = info1(1).Width;

combined = zeros(h, w, n_pairs, 'uint16');
raw_idx = 0;
pair_idx = 0;
frame_A = [];

disp('  Loading and max-projecting...');
for f = 1:numel(tif_files)
    fpath = fullfile(tif_files(f).folder, tif_files(f).name);
    info = imfinfo(fpath);
    n_frames = numel(info);
    disp(['    ', tif_files(f).name, ': ', num2str(n_frames), ' frames']);

    for t = 1:n_frames
        raw_idx = raw_idx + 1;
        fr = imread(fpath, 'Index', t);

        if mod(raw_idx, 2) == 1
            % Odd frame = plane A: store temporarily
            frame_A = uint16(fr);
        else
            % Even frame = plane B: max-project with stored plane A
            pair_idx = pair_idx + 1;
            if pair_idx <= n_pairs
                combined(:, :, pair_idx) = max(frame_A, uint16(fr));
            end
        end
    end

    disp(['      Raw frames so far: ', num2str(raw_idx), ...
          '  Pairs completed: ', num2str(pair_idx)]);
end

disp(['  Final combined movie: ', num2str(size(combined, 1)), 'x', ...
      num2str(size(combined, 2)), 'x', num2str(size(combined, 3))]);

% Save as multi-frame TIF
out_path = fullfile(output_folder, 'combined_maxproj.tif');
disp(['  Saving: ', out_path]);
for t = 1:size(combined, 3)
    if t == 1
        imwrite(combined(:,:,t), out_path, 'Compression', 'none');
    else
        imwrite(combined(:,:,t), out_path, 'Compression', 'none', 'WriteMode', 'append');
    end
end

% Also save as .mat for faster re-loading in Step 2
mat_path = fullfile(output_folder, 'combined_maxproj.mat');
disp(['  Saving MAT: ', mat_path]);
save(mat_path, 'combined', 'total_raw', 'n_pairs', 'h', 'w', '-v7.3');

disp('===== Step 1 complete =====');
end
