import os
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from numba import jit,cuda
from multiprocessing import Pool
import time
from sklearn.metrics.pairwise import manhattan_distances
path = '/home/pavgoust/workspace/'

#ray.init()


# @jit(nopython = True)
def retrieval(queries, cand):
    if os.path.exists(path + 'features_binary/{}/slides_orig.npz'.format(cand)):
        test = np.load(path + 'features_binary/{}/slides_orig.npz'.format(cand))['features']
        if test.shape[0]:
            sim = []
            for i in range(len(queries)):
                query = queries[i]

                manhattan = manhattan_distances(query, test)
                nearest = np.argmin(manhattan, axis=0)
                counts = np.zeros(nearest.shape)

                for i in range(nearest.shape[0]):
                    if i - nearest[i] >= 0:
                        counts[int(i - nearest[i])] += 1

                    for j in range(3):
                        if i - nearest[i] - j >= 0:
                            counts[int(i - nearest[i] - j)] += 1
                        if 0 <= i - nearest[i] + j < counts.shape[0]:
                            counts[int(i - nearest[i] + j)] += 1

                sim.append(max(counts))
            return sim
        else:
            print(cand)
            return None
    else:
        print(cand)
        return None



if __name__ == "__main__":


    start_time = time.time()
    with open('annotation_small.json') as f:
        annot = json.load(f)

    #keys_with_no_sound = open('../queries_with_no_sound.txt', "r").read().splitlines()

    #for key in keys_with_no_sound:
        #del annot[key]

    with open('dataset_ids.txt') as f:
        video_ids = f.readlines()

    video_ids = [x.strip() for x in video_ids]

    print(len(video_ids))
    queries = []
    queries_ids = []
    query_size = []
    for query in annot:
        temp = np.load(path+'features_binary/{}/slides_orig.npz'.format(query))['features']
        queries.append(temp)
        queries_ids.append(query)
        query_size.append(temp.shape[0])
        #print(temp.shape)

    print(len(queries_ids))

    pool = Pool()
    sims = dict()
    for cand in video_ids:
        sims[cand] = pool.apply_async(retrieval, args=[queries, cand])

    similarities = dict({query: dict() for query in queries_ids})
    for k, v in tqdm(sims.items(), total=len(sims)):
        if v.get() is not None:
            for i, s in enumerate(v.get()):
                similarities[queries_ids[i]][k] = s
    pool.terminate()
    elapsed_time = time.time() - start_time
    # np.save("slides_similarity", similarity)

    with open('similarities_orig.json', 'w') as f:
        json.dump(similarities, f)

    file = open("similarity_data.txt", "w")
    string = "Time elapsed = " + str(elapsed_time)
    file.write(string)
    file.close()
