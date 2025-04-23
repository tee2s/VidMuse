import os
import json
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import soundfile as sf

# Paths and constants
wav_dir       = '/work/users/t/i/tis/V2Music/preprocessing/bg_audio/mdx_extra'
mp4_dir       = '/work/users/t/i/tis/V2Music/preprocessing/data/video'
output_file   = './egs/V2M20K/data2.jsonl'
NUM_WORKERS   = int(os.environ.get("SLURM_CPUS_PER_TASK"))

print(f"Started building dataset with {NUM_WORKERS}")

# Ensure output dir exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def process_wav(wav_file):
    if not wav_file.endswith('.wav'):
        return None

    # derive ID and absolute paths
    ytb_id       = os.path.basename(wav_file).rsplit('_no_vocals.wav', 1)[0]
    wav_path     = os.path.join(wav_dir, wav_file)
    mp4_path     = os.path.join(mp4_dir, f"{ytb_id}.mp4")
    wav_abs, mp4_abs = os.path.abspath(wav_path), os.path.abspath(mp4_path)

    # skip missing files
    if not os.path.isfile(wav_abs) or not os.path.isfile(mp4_abs):
        return None

    try:
        info     = sf.info(wav_abs)
        duration = info.frames / info.samplerate
        sr       = info.samplerate

        return {
            "path":       wav_abs,
            "video_path": mp4_abs,
            "duration":   duration,
            "sample_rate":sr,
            "amplitude":  None,
            "weight":     None,
            "info_path":  None
        }
    except Exception:
        return None

# Kick off parallel jobs
wav_list = glob.glob(os.path.join(wav_dir, '*.wav'))
results  = []
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as exe:
    futures = {exe.submit(process_wav, f): f for f in wav_list}
    for idx, fut in enumerate(as_completed(futures), start=1):
        rec = fut.result()
        src = futures[fut]
        if rec:
            results.append(rec)
            print(f"[{idx}/{len(wav_list)}] ✓ {os.path.basename(rec['path'])} — {rec['duration']:.2f}s")
        else:
            print(f"[{idx}/{len(wav_list)}] ✗ skipped {src}")

# Write out once
with open(output_file, 'w') as f_out:
    for rec in results:
        f_out.write(json.dumps(rec) + '\n')
print(f"Done — wrote {len(results)} records to {output_file}")