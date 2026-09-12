function out = pain_2plane_step1_split_planes(session_dir, output_folder)
% Split the interleaved ETL depths into two separate movies.
%
% REPLACES pain_2plane_step1_merge_planes.m, which max-projected the depth
% pair into one frame. That was measured to be wrong for this data
% (EXTRACT_QC_20260911.md):
%
%   * the two depths (65 um and 15 um) hold largely DIFFERENT somata -
%     17 at one depth, 14 at the other, and 5 of the 17 land within one
%     cell radius of a cell at the other depth, so max-projection merges
%     roughly a third of them into single ROIs;
%   * max() is nonlinear and neither depth dominates (one wins 49 % of
%     pixel-frames), so the projected value flips between the two neurons
%     in 50 % of consecutive frames. Half of every projected trace was
%     two cells swapping, which no EXTRACT parameter can undo.
%
% Splitting keeps each depth as its own 4.605 Hz movie, which is the rate
% the rest of the pipeline already assumes. Cells then carry a plane label
% and every trace belongs to one neuron.
%
% Raw frame parity follows the original convention: odd raw frames are
% plane A, even are plane B. Which physical depth that is (65 vs 15 um)
% depends on the ETL scan direction in etl1data.csv - Descending, so
% plane A is the FIRST listed position (65 um). Recorded in out.note
% rather than assumed anywhere downstream.
%
% Usage:
%   out = pain_2plane_step1_split_planes(session_dir)
%   out = pain_2plane_step1_split_planes(session_dir, output_folder)
%
% Returns struct with .plane_A_tif, .plane_B_tif, .n_frames_per_plane.

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

% natural numeric order: CellVideo 1.tif, CellVideo 2.tif, ...
names = {tif_files.name};
nums = zeros(1, numel(names));
for i = 1:numel(names)
    tok = regexp(names{i}, '(\d+)', 'tokens');
    if ~isempty(tok), nums(i) = str2double(tok{end}{1}); end
end
[~, si] = sort(nums);
tif_files = tif_files(si);

disp('===== Step 1: split interleaved ETL depths =====');
disp(['  TIF folder : ', tif_dir]);
disp(['  Files      : ', num2str(numel(tif_files))]);

total_raw = 0;
for f = 1:numel(tif_files)
    total_raw = total_raw + numel(imfinfo(fullfile( ...
        tif_files(f).folder, tif_files(f).name)));
end
if mod(total_raw, 2) ~= 0
    warning('Odd raw frame count (%d); the last frame is dropped so the two planes stay equal length.', total_raw);
end
n_per_plane = floor(total_raw / 2);
info1 = imfinfo(fullfile(tif_files(1).folder, tif_files(1).name));
h = info1(1).Height;
w = info1(1).Width;
disp(['  Raw frames : ', num2str(total_raw)]);
disp(['  Per plane  : ', num2str(n_per_plane), '  (', num2str(h), 'x', num2str(w), ')']);

A = zeros(h, w, n_per_plane, 'uint16');
B = zeros(h, w, n_per_plane, 'uint16');

raw_idx = 0; ia = 0; ib = 0;
t0 = tic;
for f = 1:numel(tif_files)
    fpath = fullfile(tif_files(f).folder, tif_files(f).name);
    nfr = numel(imfinfo(fpath));
    % one file handle for the whole file: imread reopens and re-parses the
    % directory on every call, which on a ~900 MB stack costs minutes
    tobj = Tiff(fpath, 'r');
    for t = 1:nfr
        raw_idx = raw_idx + 1;
        fr = tobj.read();
        if mod(raw_idx, 2) == 1
            if ia < n_per_plane, ia = ia + 1; A(:, :, ia) = fr; end
        else
            if ib < n_per_plane, ib = ib + 1; B(:, :, ib) = fr; end
        end
        if t < nfr, tobj.nextDirectory(); end
    end
    tobj.close();
    disp(['    ', tif_files(f).name, ': ', num2str(nfr), ' frames  ', ...
          '(A=', num2str(ia), ' B=', num2str(ib), ')  ', ...
          num2str(toc(t0), '%.0f'), ' s']);
end

pathA = fullfile(output_folder, 'plane_A.tif');
pathB = fullfile(output_folder, 'plane_B.tif');
write_stack(A, pathA);
write_stack(B, pathB);

% QC: mean image per plane and how different they are, so the split can be
% eyeballed without loading the movies again
mA = mean(single(A), 3);
mB = mean(single(B), 3);
r = corr(mA(:), mB(:));
save(fullfile(output_folder, 'plane_split_qc.mat'), 'mA', 'mB', 'r', ...
     'n_per_plane', 'total_raw', '-v7.3');
try
    fh = figure('Visible', 'off', 'Position', [100 100 1200 520]);
    subplot(1,2,1); imagesc(mA); axis image off; colormap gray;
    title(sprintf('plane A mean (%d frames)', n_per_plane));
    subplot(1,2,2); imagesc(mB); axis image off; colormap gray;
    title(sprintf('plane B mean   corr(A,B) = %.3f', r));
    exportgraphics(fh, fullfile(output_folder, 'plane_split_qc.png'), ...
                   'Resolution', 140);
    close(fh);
catch ME
    warning('QC figure failed: %s', ME.message);
end

disp(['  corr(mean A, mean B) = ', num2str(r, '%.3f'), ...
      '   (well below 1 confirms the depths differ)']);
disp(['  wrote ', pathA]);
disp(['  wrote ', pathB]);
disp('===== Step 1 complete =====');

out = struct('plane_A_tif', pathA, 'plane_B_tif', pathB, ...
             'n_frames_per_plane', n_per_plane, 'corr_AB', r, ...
             'note', ['odd raw frames = plane A; etl1data.csv is ' ...
                      'Descending so plane A is the first listed ETL ' ...
                      'position (65 um)']);
end


function write_stack(M, path)
% Write a uint16 stack through one Tiff handle. imwrite with WriteMode
% append reopens the file per frame and takes minutes for 3000 frames.
if isfile(path), delete(path); end
[h, w, n] = size(M);
t = Tiff(path, 'w8');          % BigTIFF: 3000 x 440 x 512 x 2 B > 1.3 GB
tags = struct('ImageLength', h, 'ImageWidth', w, ...
              'Photometric', Tiff.Photometric.MinIsBlack, ...
              'BitsPerSample', 16, 'SamplesPerPixel', 1, ...
              'RowsPerStrip', 16, ...
              'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
              'Compression', Tiff.Compression.None);
for k = 1:n
    t.setTag(tags);
    t.write(M(:, :, k));
    if k < n, t.writeDirectory(); end
end
t.close();
end
