import os
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from numba import jit,cuda
from multiprocessing import Pool
import time
from scipy.sparse import csr_matrix

path = '/home/pavgoust/workspace/'


def to_binary(test):
    binary = csr_matrix((np.ones((test.shape[0]*24)), (np.arange(test.shape[0]).repeat(24),
                                                         test.flatten())), shape=(test.shape[0], 850))
    return binary


# @jit(nopython = True, parallel = True)
def retrieval(queries, cand):
    if os.path.exists(path + 'features_binary/{}/tiles_orig.npz'.format(cand)):
        test = to_binary(np.load(path + 'features_binary/{}/tiles_orig.npz'.format(cand))['features']).T
        if test.shape[0]:
            sim = []
            for query in queries:
                s = query.dot(test).toarray()
                nearest = np.argmax(s, axis=0)
                counts = np.zeros(nearest.shape)

                for i in range(nearest.shape[0]):
                    if i-nearest[i]>=0:

                        counts[int(i-nearest[i])] += 1

                    for j in range(3):
                        if i - nearest[i]-j >= 0:
                            counts[int(i - nearest[i]-j)] += 1
                        if i - nearest[i] + j >= 0 and i - nearest[i] + j<counts.shape[0]:
                            counts[int(i - nearest[i] + j)] += 1

                sim.append(max(counts))
            return sim
        else:
            print(cand)
            return None
    else:
        print(cand)
        return None

'''
    start_query_part = query.shape[0] - 4
    end_query_part = query.shape[0]
    start_test_part = 0
    pointer = 4
    end_test_part = min(test.shape[0], pointer)

    counts = []
    iterations = test.shape[0]+query.shape[0]-6-1
    flag = 1
    for i in range(iterations):

        query_part = np.asarray(list(range(start_query_part, end_query_part)))
        test_part1 = nearest[start_test_part:end_test_part]
        test_part2 = test_part1-1
        test_part3 = test_part1+1
        test_part4 = test_part1-2
        test_part5 = test_part1+2
        test_part6 = test_part1-3
        test_part7 = test_part1+3
        #print(str(query_part.shape) + "  " + str(test_part.shape) )
        counts1 = sum(query_part == test_part1)
        counts2 = sum(query_part == test_part2)
        counts3 = sum(query_part == test_part3)
        counts4 = sum(query_part == test_part4)
        counts5 = sum(query_part == test_part5)
        counts6 = sum(query_part == test_part6)
        counts7 = sum(query_part == test_part7)
        counts.append(counts1+counts2+counts3+counts4+counts5+counts6+counts7)

        pointer += 1

        start_query_part = max(start_query_part-1, 0)

        if pointer > test.shape[0]:
            end_query_part = end_query_part-1

        if start_query_part == 0 and flag == 0:
            start_test_part = start_test_part+1

        if start_query_part == 0:
            flag=0

        end_test_part = min(test.shape[0], pointer)
'''




if __name__ == "__main__":

    start_time = time.time()
    with open('../annotation_small.json') as f:
        annot = json.load(f)

    with open('../dataset_ids.txt') as f:
        video_ids = f.readlines()

    video_ids = [x.strip() for x in video_ids]

    print(len(video_ids))
    queries = []
    queries_ids = []
    query_size = []
    for query in annot:
        temp = to_binary(np.load(path + 'features_binary/{}/tiles_orig.npz'.format(query))['features'])
        queries.append(temp)
        queries_ids.append(query)
        query_size.append(temp.shape[0])
        # print(temp.shape)

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

    with open('similarities_orig_rev.json', 'w') as f:
        json.dump(similarities, f, indent=1)

    file = open("similarity_data.txt", "w")
    string = "Time elapsed = " + str(elapsed_time)
    file.write(string)
    file.close()
