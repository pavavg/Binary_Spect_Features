import os
import wave
import librosa as lib
import numpy as np
import subprocess as sp
import json
from tqdm import tqdm
from multiprocessing import Pool
import random
import time



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
            # print(binary.shape)

            feature_vector = np.zeros(48)
            for i in range(24):
                hor_slice = binary[14 * i:min(14 * (i + 1), 332), :]

                ver_slice = binary[:, 11 * i:min(11 * (i + 1), 256)]
                # print(hor_slice.shape)
                # print(ver_slice.shape)

                feature_vector[i] = np.sum(hor_slice)
                feature_vector[24 + i] = np.sum(ver_slice)

            #print(feature_vector)
            features.append(feature_vector)
            step = step + 8    # 8 * 3(hop_size) ms = 24 ms frame advance

        features = np.array(features)
        #print(features.shape)
        if features.shape[0]:
            np.savez_compressed('{}/{}/slides_orig'.format(feature_dir, v), features=features)



n_fft = 768         # 96 ms
hop_length = 24     # 3 ms
video_dir = '/home/pavgoust/fivr/videos'

with open('dataset_ids.txt') as f:
    video_ids = f.readlines()

video_ids = [x.strip() for x in video_ids] # list of video ids

random.shuffle(video_ids)
with open('annotation_small.json', 'r') as f:
    annot = json.load(f)

for query in annot:
    video_ids.append(query)

feature_dir = '/home/pavgoust/workspace/features_binary'


pbar = tqdm(video_ids)  # progress bar

# multithreading spectrogram extraction
pool = Pool(8)

f = open('corrupted.txt','a')


for v in video_ids:

    if os.path.exists('{}/{}/slides_orig.npz'.format(feature_dir, v)):

        pbar.update()
        continue

    pool.apply_async(feat_extract, args=[v], callback=lambda *a: pbar.update())
    f.writelines(v+'\n')
f.close()
pool.close()
pool.join()
pool.terminate()

