import json
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from numba import jit,cuda
from multiprocessing import Pool
import time
path = '/home/pavgoust/workspace/'


@jit(nopython = True)
def retrieval(query, test):
    nearest = np.zeros(test.shape[0])
    #print(nearest.shape)
    #print(test.shape)
    for i in range(test.shape[0]):

        manhattan = np.zeros(query.shape[0])
        for j in range(query.shape[0]):
            manhattan[j] = np.sum(np.abs(np.subtract(query[j, :], test[i, :]) ) )
        nearest[i] = np.argmin(manhattan)
        #nearest[i] = (manhattan.index(min(manhattan)))
    #nearest = np.asarray(nearest)

    counts = np.zeros(nearest.shape)

    for i in range(nearest.shape[0]):
        if i-nearest[i]>=0:

            counts[int(i-nearest[i])] += 1

        for j in range(3):
            if i - nearest[i]-j >= 0:
                counts[int(i - nearest[i]-j)] += 1
            if i - nearest[i] + j >= 0 and i - nearest[i] + j<counts.shape[0]:
                counts[int(i - nearest[i] + j)] += 1

    return max(counts)


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

    print(min(query_size))

    #similarity = []
    similarity = np.empty([len(queries), len(video_ids)])
    cand_size = []

    pool = Pool(4)

    similarities = dict({query: dict() for query in queries_ids})
    j=0
    for cand in tqdm(video_ids):
        #print(cand)
        #try:
        cand_feat = np.load(path+'features_binary/{}/slides_orig.npz'.format(cand))['features']
        if cand_feat.shape[0]:
            cand_size.append(cand_feat.shape[0])
            i=0
            temp_list = []
            for i in range(len(queries)) :
                query_feat = queries[i]

                similarity[i,j] = pool.apply(retrieval, args=[query_feat,cand_feat])

                similarities[queries_ids[i]][cand] = similarity[i,j]
                i = i+1

        else:
            print(cand)
        j = j+1


    elapsed_time = time.time() - start_time
    #similarity = np.asarray(similarity)
    np.save("slides_similarity", similarity)

    with open('similarities_orig.json', 'w') as f:
        json.dump(similarities, f)

    file = open("similarity_data.txt", "w")
    string = "Time elapsed = " + str(elapsed_time)
    file.write(string)
    file.close()
