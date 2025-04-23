import os
import json
import librosa

wav_dir = './dataset/V2M20K/train/audio'
mp4_dir = './dataset/V2M20K/train/video'

output_file = './egs/V2M20K/train/data.jsonl'
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
count = 0
with open(output_file, 'w') as f_out:
    for wav_file in os.listdir(wav_dir):
        # only process .wav files
        if not wav_file.endswith('.wav'):
            continue

        # derive ID and absolute paths
        ytb_id = wav_file.rsplit('_no_vocals.wav', 1)[0]
        wav_path_abs = os.path.abspath(os.path.join(wav_dir, wav_file))
        mp4_path_abs = os.path.abspath(os.path.join(mp4_dir, f"{ytb_id}.mp4"))

        # skip if either file is missing on disk
        if not os.path.isfile(wav_path_abs):
            print(f"Skipping {wav_file}: WAV not found at {wav_path_abs}")
            continue
        if not os.path.isfile(mp4_path_abs):
            print(f"Skipping {wav_file}: MP4 not found at {mp4_path_abs}")
            continue

        try:
            # load audio and measure duration
            audio, sr = librosa.load(wav_path_abs, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)

            # build info record
            info_dict = {
                "path": wav_path_abs,
                "video_path": mp4_path_abs,
                "duration": duration,
                "sample_rate": sr,
                "amplitude": None,
                "weight": None,
                "info_path": None
            }

            count += 1
            print(f"Successfully processed File {count}: {ytb_id} ({duration:.2f}s)")
            f_out.write(json.dumps(info_dict) + '\n')

        except Exception as e:
            print(f"Skipping {wav_file}: {e}")
            continue