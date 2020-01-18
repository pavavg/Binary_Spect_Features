import os
import wave
import librosa as lib
import numpy as np
import subprocess as sp
import json
from tqdm import tqdm
from multiprocessing import Pool
import time


n_fft = 768         # 96 ms
hop_length = 24     # 6 ms
video_dir = '/home/pavgoust/fivr/videos'


def ffmpeg_load_audio(filename, sr=8000, mono=True, normalize=True, in_type=np.int16, out_type=np.float32):
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]

    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']

    p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=4096)
    # amy-uprint(filename)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr  # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.frombuffer(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            audio -= np.mean(audio)
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max

    return audio, sr


def extract_spectrogram(video_path):
    #if os.path.exists(video_path):
     #   return np.array([])

    try:
        audio, sr = ffmpeg_load_audio('{}/{}/video.mp4'.format(video_dir, video_path))
        #print(len(audio))
        spectgram = lib.feature.melspectrogram(
            y=audio, sr=8000, n_fft=n_fft, hop_length=hop_length, n_mels=257, fmin=500, fmax=3000)
        spectgram = spectgram.T
        spectgram = np.concatenate([spectgram, np.zeros((333 - spectgram.shape[0] % 333, 257))])  # zero padding
        #np.savez_compressed('/home/pavgoust/workspace/features/{}/spectgram'.format(video_path), spectgram=spectgram)
        return spectgram
    except Exception as e:
        print(e)
        return np.array([])



path = '/home/pavgoust/fivr'
with open('dataset_ids.txt') as f:
    video_ids = f.readlines()

video_ids = [x.strip() for x in video_ids] # list of video ids


feature_dir = '/home/pavgoust/workspace/features_binary'


def feat_extract(v):

    inpt = extract_spectrogram(v)

    if inpt.shape[0]:
        # print(inpt.shape)

        if not os.path.exists('{}/{}'.format(feature_dir, v)):
            os.makedirs('{}/{}'.format(feature_dir, v))

        features = []
        step = 0
        while step + 333 < inpt.shape[0]:
            tempSpec = inpt[step:step + 333, :]
            mean = np.mean(tempSpec)
            binary = tempSpec > mean
            binary = binary.astype(np.int)
            #print(binary.shape)

            hor_tiles = int((257 ) // 10)
            ver_tiles = int((333 ) // 10)
            tile_size = 10
            tiles_sum = np.zeros(ver_tiles*hor_tiles)

            for i in range(ver_tiles):

                start_row = i*tile_size

                end_row = min(start_row+tile_size, 333)

                for j in range(hor_tiles):

                    start_col = j*tile_size
                    end_col = min(start_col+tile_size, 257)
                    tile = binary[start_row: end_row, start_col: end_col]
                    #print(tile.sum())
                    tiles_sum[i*hor_tiles + j] = tile.sum()

            feature_vector = np.argsort(tiles_sum)[::-1][0:24]
            #print(feature_vector)
            features.append(feature_vector)
            step = step + 40    # 40 * 3(hop_size) ms = 120 ms frame advance

        features = np.array(features)

        if features.shape[0]:
            np.savez_compressed('{}/{}/tiles_orig'.format(feature_dir, v), features=features)


with open('annotation_small.json', 'r') as f:
    annot = json.load(f)

for query in annot:
    video_ids.append(query)

pbar = tqdm(video_ids)  # progress bar

# multithreading spectrogram extraction
pool = Pool(8)

#future = dict()
#for v in video_ids:
    # extract_spectrogram(v)
#    if not os.path.exists('{}/{}/binary.npz'.format(feature_dir, v)):
#        future[v] = pool.apply_async(extract_spectrogram, args=[v])

for v in video_ids:

    #if v in future:
    #if os.path.exists('{}/{}/tiles.npz'.format(feature_dir, v)):
        #del future[v]
       # pbar.update()
       # continue
    #feat_extract(v)
    pool.apply_async(feat_extract, args=[v], callback=lambda *a: pbar.update())

pool.close()
pool.join()
pool.terminate()

