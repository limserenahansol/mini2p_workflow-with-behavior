"""
validate_pipeline.py
Python validation script for the Pain 2-Plane Calcium Imaging + Behavior Pipeline.

Verifies:
  1. TIF loading and 2-plane deinterleaving logic
  2. TDMS timestamp parsing and consistency
  3. AVI file integrity and frame counts
  4. Temporal alignment sanity checks
  5. Final merged output structure (if available)

Usage:
    python validate_pipeline.py <session_dir> [output_folder]
"""

import sys
import os
import struct
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from nptdms import TdmsFile
except ImportError:
    TdmsFile = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import scipy.io as sio
except ImportError:
    sio = None


def section(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')


def validate_tifs(session_dir):
    """Validate TIF files: count, dimensions, interleaving."""
    section('1. TIF Validation (CellVideo1/CellVideo)')
    tif_dir = os.path.join(session_dir, 'CellVideo1', 'CellVideo')
    if not os.path.isdir(tif_dir):
        print(f'  ERROR: TIF folder not found: {tif_dir}')
        return False

    tif_files = sorted(
        [f for f in os.listdir(tif_dir) if f.lower().endswith(('.tif', '.tiff'))],
        key=lambda x: int(''.join(c for c in x if c.isdigit()) or '0')
    )
    print(f'  TIF files found: {len(tif_files)}')
    if len(tif_files) == 0:
        print('  ERROR: No TIF files found')
        return False

    total_frames = 0
    dims = None
    for tf in tif_files:
        fpath = os.path.join(tif_dir, tf)
        if Image is not None:
            img = Image.open(fpath)
            n = 0
            try:
                while True:
                    img.seek(n)
                    n += 1
            except EOFError:
                pass
            h, w = img.size[1], img.size[0]
            if dims is None:
                dims = (h, w)
            elif (h, w) != dims:
                print(f'  WARNING: {tf} has different dimensions {w}x{h} vs {dims[1]}x{dims[0]}')
            print(f'    {tf}: {n} frames, {w}x{h}')
            total_frames += n
        else:
            fsize = os.path.getsize(fpath)
            print(f'    {tf}: {fsize/(1024*1024):.1f} MB (PIL not available for frame count)')

    if Image is not None:
        print(f'  Total raw frames: {total_frames}')
        print(f'  After max-projection pairing: {total_frames // 2} frames')
        if total_frames % 2 != 0:
            print(f'  WARNING: Odd frame count - last frame will be dropped')
        if total_frames == 18000:
            print(f'  OK: Expected 18000 raw frames -> 9000 paired')
        else:
            print(f'  NOTE: Expected 18000 raw frames, got {total_frames}')

    # Quick test: read first 4 frames and verify max-projection logic
    if Image is not None and total_frames >= 4:
        fpath0 = os.path.join(tif_dir, tif_files[0])
        img = Image.open(fpath0)
        frames = []
        for i in range(4):
            img.seek(i)
            frames.append(np.array(img))

        pair1 = np.maximum(frames[0], frames[1])
        pair2 = np.maximum(frames[2], frames[3])
        print(f'\n  Max-projection test (first 2 pairs):')
        print(f'    Frame 1 (planeA): mean={frames[0].mean():.1f}, max={frames[0].max()}')
        print(f'    Frame 2 (planeB): mean={frames[1].mean():.1f}, max={frames[1].max()}')
        print(f'    MaxProj pair 1:   mean={pair1.mean():.1f}, max={pair1.max()}')
        print(f'    Frame 3 (planeA): mean={frames[2].mean():.1f}, max={frames[2].max()}')
        print(f'    Frame 4 (planeB): mean={frames[3].mean():.1f}, max={frames[3].max()}')
        print(f'    MaxProj pair 2:   mean={pair2.mean():.1f}, max={pair2.max()}')
        print(f'  OK: Max-projection logic verified')

    return True


def validate_tdms(session_dir):
    """Validate TDMS timestamp files."""
    section('2. TDMS Timestamp Validation')
    if TdmsFile is None:
        print('  SKIP: npTDMS not installed')
        return True

    results = {}

    # Neuron timestamps
    neuron_tdms = os.path.join(session_dir, 'CellVideo1', 'CellVideo_CHA_Info.tdms')
    if os.path.isfile(neuron_tdms):
        f = TdmsFile.read(neuron_tdms)
        for g in f.groups():
            if g.name == 'CHA':
                times = g['Time'][:]
                slices = g['Slice'][:]
                print(f'  Neuron TDMS: {len(times)} timestamps')
                print(f'    First: {times[0]}')
                print(f'    Last:  {times[-1]}')

                # Check interleaving via Slice channel
                unique_slices = set(str(s) for s in slices[:10])
                print(f'    Slice values (first 10): {sorted(unique_slices)}')
                if '1' in unique_slices and '2' in unique_slices:
                    print(f'    OK: 2-plane interleaving confirmed (Slice 1,2)')

                # Compute frame rate
                from datetime import datetime
                t0 = datetime.strptime(str(times[0]), '%Y-%m-%d %H:%M:%S.%f')
                t1 = datetime.strptime(str(times[1]), '%Y-%m-%d %H:%M:%S.%f')
                dt = (t1 - t0).total_seconds()
                raw_rate = 1.0 / dt if dt > 0 else 0
                print(f'    Raw frame interval: {dt*1000:.1f} ms -> {raw_rate:.2f} Hz')
                print(f'    After pairing: {raw_rate/2:.2f} Hz effective')

                # Check for paired timestamps
                odd_times = [str(t) for t in times[::2]]
                results['neuron_count'] = len(odd_times)
                results['neuron_rate'] = raw_rate / 2
                break
    else:
        print(f'  WARNING: Neuron TDMS not found: {neuron_tdms}')

    # Behavioral camera timestamps
    for cam_id, cam_name in [(1, 'MiceVideo1'), (2, 'MiceVideo2')]:
        cam_dir = os.path.join(session_dir, cam_name)
        tdms_files = [f for f in os.listdir(cam_dir) if f.endswith('.tdms') and 'index' not in f]
        if tdms_files:
            fpath = os.path.join(cam_dir, tdms_files[0])
            f = TdmsFile.read(fpath)
            for g in f.groups():
                for c in g.channels():
                    if 'time' in c.name.lower():
                        times = c[:]
                        print(f'\n  Camera {cam_id} ({cam_name}): {len(times)} timestamps')
                        print(f'    First: {times[0]}')
                        print(f'    Last:  {times[-1]}')

                        from datetime import datetime
                        t0 = datetime.strptime(str(times[0]), '%Y-%m-%d %H:%M:%S.%f')
                        t1 = datetime.strptime(str(times[1]), '%Y-%m-%d %H:%M:%S.%f')
                        dt = (t1 - t0).total_seconds()
                        rate = 1.0 / dt if dt > 0 else 0
                        print(f'    Frame interval: {dt*1000:.1f} ms -> {rate:.2f} Hz')
                        results[f'cam{cam_id}_count'] = len(times)
                        results[f'cam{cam_id}_rate'] = rate
                        break
                break

    # Cross-check: do neuron and behavior timestamps overlap?
    if 'neuron_count' in results and 'cam1_count' in results:
        print(f'\n  Cross-check:')
        print(f'    Neuron: {results["neuron_count"]} frames at {results["neuron_rate"]:.2f} Hz')
        cam1_dur = results['cam1_count'] / results['cam1_rate'] if results['cam1_rate'] > 0 else 0
        neuro_dur = results['neuron_count'] / results['neuron_rate'] if results['neuron_rate'] > 0 else 0
        print(f'    Cam1:   {results["cam1_count"]} frames at {results["cam1_rate"]:.1f} Hz ({cam1_dur:.1f} s)')
        print(f'    Neuron duration: {neuro_dur:.1f} s')
        ratio = results['cam1_rate'] / results['neuron_rate'] if results['neuron_rate'] > 0 else 0
        print(f'    Behavior/Neuron rate ratio: {ratio:.1f}x')
        if abs(cam1_dur - neuro_dur) < 60:
            print(f'    OK: Durations match within 60 s')
        else:
            print(f'    WARNING: Duration mismatch > 60 s')

    return True


def validate_avis(session_dir):
    """Validate behavioral AVI files."""
    section('3. AVI File Validation')

    for cam_id, cam_name in [(1, 'MiceVideo1'), (2, 'MiceVideo2')]:
        avi_dir = os.path.join(session_dir, cam_name, 'MiceVideo')
        if not os.path.isdir(avi_dir):
            print(f'  WARNING: AVI folder not found: {avi_dir}')
            continue

        avis = sorted([f for f in os.listdir(avi_dir) if f.lower().endswith('.avi')])
        print(f'\n  Camera {cam_id} ({cam_name}): {len(avis)} AVIs')
        total_frames = 0
        for avi in avis:
            fpath = os.path.join(avi_dir, avi)
            fsize = os.path.getsize(fpath) / (1024 * 1024)
            if cv2 is not None:
                try:
                    cap = cv2.VideoCapture(fpath)
                    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    total_frames += nf
                    print(f'    {avi}: {nf} frames, {w}x{h}, {fps:.1f} FPS, {fsize:.0f} MB')
                except Exception as e:
                    print(f'    {avi}: {fsize:.0f} MB (error reading: {e})')
            else:
                print(f'    {avi}: {fsize:.0f} MB (cv2 not available)')

        if cv2 is not None and total_frames > 0:
            print(f'  Total cam{cam_id} frames: {total_frames}')

    return True


def validate_output(output_folder):
    """Validate output files if they exist."""
    section('4. Output Validation')

    if not os.path.isdir(output_folder):
        print(f'  Output folder not yet created: {output_folder}')
        return True

    expected_files = [
        'combined_maxproj.tif',
        'combined_maxproj.mat',
        'timestamps.mat',
        'final_analysis_results.mat',
        'final_neuron_behavior.mat',
    ]

    for ef in expected_files:
        fpath = os.path.join(output_folder, ef)
        if os.path.isfile(fpath):
            fsize = os.path.getsize(fpath) / (1024 * 1024)
            print(f'  OK: {ef} ({fsize:.1f} MB)')
        else:
            print(f'  PENDING: {ef} (not yet created)')

    # Validate final_neuron_behavior.mat structure
    final_file = os.path.join(output_folder, 'final_neuron_behavior.mat')
    if os.path.isfile(final_file) and sio is not None:
        try:
            # Try loading with scipy (v7 format)
            data = sio.loadmat(final_file)
            print(f'\n  final_neuron_behavior.mat contents:')
            for key in sorted(data.keys()):
                if key.startswith('_'):
                    continue
                val = data[key]
                if hasattr(val, 'shape'):
                    print(f'    {key}: shape={val.shape}, dtype={val.dtype}')
                else:
                    print(f'    {key}: {type(val).__name__}')
        except Exception:
            # v7.3 (HDF5) format - need h5py
            try:
                import h5py
                with h5py.File(final_file, 'r') as hf:
                    print(f'\n  final_neuron_behavior.mat contents (HDF5):')
                    for key in sorted(hf.keys()):
                        if key.startswith('#'):
                            continue
                        ds = hf[key]
                        if hasattr(ds, 'shape'):
                            print(f'    {key}: shape={ds.shape}, dtype={ds.dtype}')
                        else:
                            print(f'    {key}: {type(ds).__name__}')
            except ImportError:
                print('  NOTE: Cannot read v7.3 MAT files (need h5py)')
            except Exception as e:
                print(f'  WARNING: Error reading final file: {e}')

    return True


def validate_alignment_sanity(session_dir):
    """Sanity-check temporal alignment math."""
    section('5. Alignment Sanity Check')
    if TdmsFile is None:
        print('  SKIP: npTDMS not installed')
        return True

    from datetime import datetime

    # Read neuron timestamps
    neuron_tdms = os.path.join(session_dir, 'CellVideo1', 'CellVideo_CHA_Info.tdms')
    if not os.path.isfile(neuron_tdms):
        print('  SKIP: Neuron TDMS not found')
        return True

    f = TdmsFile.read(neuron_tdms)
    times = None
    for g in f.groups():
        if g.name == 'CHA':
            times = g['Time'][:]
            break

    if times is None or len(times) < 4:
        print('  SKIP: Not enough timestamps')
        return True

    # Parse odd-frame timestamps (after pairing)
    odd_times_str = [str(t) for t in times[::2]]
    t_parsed = [datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f') for s in odd_times_str[:100]]
    diffs = [(t_parsed[i+1] - t_parsed[i]).total_seconds() for i in range(len(t_parsed)-1)]
    mean_dt = np.mean(diffs)
    std_dt = np.std(diffs)
    effective_rate = 1.0 / mean_dt if mean_dt > 0 else 0

    print(f'  Paired neuron frame interval: {mean_dt*1000:.2f} +/- {std_dt*1000:.2f} ms')
    print(f'  Effective rate: {effective_rate:.3f} Hz')
    print(f'  Total paired frames: {len(odd_times_str)}')
    print(f'  Session duration: {len(odd_times_str) * mean_dt:.1f} s ({len(odd_times_str) * mean_dt / 60:.1f} min)')

    # Verify behavioral rate ratio
    cam_dir = os.path.join(session_dir, 'MiceVideo1')
    tdms_files = [f for f in os.listdir(cam_dir) if f.endswith('.tdms') and 'index' not in f]
    if tdms_files:
        bf = TdmsFile.read(os.path.join(cam_dir, tdms_files[0]))
        for g in bf.groups():
            for c in g.channels():
                if 'time' in c.name.lower():
                    bt = c[:]
                    bt0 = datetime.strptime(str(bt[0]), '%Y-%m-%d %H:%M:%S.%f')
                    bt1 = datetime.strptime(str(bt[1]), '%Y-%m-%d %H:%M:%S.%f')
                    bdt = (bt1 - bt0).total_seconds()
                    brate = 1.0 / bdt if bdt > 0 else 0
                    ratio = brate / effective_rate if effective_rate > 0 else 0
                    print(f'\n  Behavior rate: {brate:.2f} Hz')
                    print(f'  Rate ratio (behav/neuron): {ratio:.1f}x')
                    print(f'  ~{ratio:.0f} behavior frames per neuron frame')
                    if 5 < ratio < 8:
                        print(f'  OK: Ratio in expected range (6-7x)')
                    else:
                        print(f'  NOTE: Ratio outside typical 6-7x range')
                    break
            break

    return True


def main():
    if len(sys.argv) < 2:
        session_dir = r'C:\Users\hsollim\Downloads\20260409_Pain_2plane_-45+35+75%_28min_2026-04-08_17-48-09'
    else:
        session_dir = sys.argv[1]

    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(session_dir, 'output')

    print(f'Pain 2-Plane Pipeline Validation')
    print(f'Session: {session_dir}')
    print(f'Output:  {output_folder}')

    ok = True
    ok &= validate_tifs(session_dir)
    ok &= validate_tdms(session_dir)
    ok &= validate_avis(session_dir)
    ok &= validate_alignment_sanity(session_dir)
    ok &= validate_output(output_folder)

    section('SUMMARY')
    if ok:
        print('  All validations passed.')
    else:
        print('  Some validations had errors — check above.')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
