import json
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from numba import jit,cuda
from multiprocessing import Pool
import time
path = '/home/pavgoust/workspace/'



@jit(nopython = True, parallel = True)
def retrieval(query, test):
    nearest = np.zeros(test.shape[0])
    #print(nearest.shape)
    #print(test.shape)
    for i in range(test.shape[0]):

        sim = np.zeros(query.shape[0])
        for j in range(query.shape[0]):

            #sim[j] = np.intersect1d(query[j, :], test[i, :]).shape[0]
            for k in range(query.shape[1]):
                for t in range(query.shape[1]):
                    if query[j,k] == test[i,t]:
                        sim[j] +=1
                        break

            #print(sim[j])
            #manhattan[j] = np.sum(np.abs(np.subtract(query[j, :], test[i, :]) ) )
        nearest[i] = np.argmax(sim)
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
    with open('../annotation.json') as f:
        annot = json.load(f)

    keys_with_no_sound = open('../queries_with_no_sound.txt', "r").read().splitlines()

    for key in keys_with_no_sound:
        del annot[key]

    extracted_features = np.load('../extracted_features_tiles.npy')
    extracted_features = extracted_features[:100000]
    print(len(extracted_features))
    queries = []
    queries_ids = []
    query_size = []
    for query in annot:
        temp = np.load(path+'features_binary/{}/tiles.npz'.format(query))['features']
        queries.append(temp)
        queries_ids.append(query)
        query_size.append(temp.shape[0])
        #print(temp.shape)

    print(min(query_size))

    #similarity = []
    similarity = np.empty([len(queries), len(extracted_features)])
    cand_size = []

    pool = Pool(8)

    similarities = dict({query: dict() for query in queries_ids})
    j=0
    for cand in tqdm(extracted_features):
        #print(cand)
        #try:
        cand_feat = np.load(path+'features_binary/{}/tiles.npz'.format(cand))['features']
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
    np.save("tiles_similarity", similarity)

    with open('similarities.json', 'w') as f:
        json.dump(similarities, f)

    file = open("similarity_data.txt", "w")
    string = "Time elapsed = " + str(elapsed_time)
    file.write(string)
    file.close()
