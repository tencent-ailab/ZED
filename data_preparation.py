import ujson as json
from tqdm import tqdm
from util import maven_e_type_to_definition
from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class NumpySearcher(object):
    def __init__(self, reference_vectors):
        self.reference_vectors = reference_vectors

    def search(self, query_vector, final_num_neighbors=5):
        scores = list()
        for tmp_reference in self.reference_vectors:
            scores.append(np.dot(query_vector, tmp_reference))
        scores = np.asarray(scores)
        sort_index = np.argsort(scores)
        sort_index = sort_index[::-1]
        return_values = list()
        for i in sort_index[:final_num_neighbors]:
            return_values.append(scores[i])
        return sort_index[:final_num_neighbors], return_values

    def search_batch(self, query_vectors, final_num_neighbors=5):
        indexes = list()
        values = list()
        for tmp_query in query_vectors:
            tmp_indexes, tmp_values = self.search(tmp_query, final_num_neighbors)
            indexes.append(tmp_indexes)
            values.append(tmp_values)
        return indexes, values


# We can set the number of instances per wordnet gloss. By default, we set it to be 10
NUMBERINSTANCEPERTYPE = 10
MAX_LENGTH = 512

with open('final_aligned_data.json', 'r', encoding='utf-8') as f:
    all_paired_data = json.load(f)
print('finish loading training instances...')

gloss2instance = dict()
for tmp_example in tqdm(all_paired_data, desc='Loading paired data'):
    if tmp_example['gloss'] not in gloss2instance:
        gloss2instance[tmp_example['gloss']] = list()
    gloss2instance[tmp_example['gloss']].append(tmp_example)

pretrain_paired_data = list()
selected_glosses = list()
for tmp_gloss in gloss2instance:
    TMPCOUNT = 0
    for tmp_instance in gloss2instance[tmp_gloss]:
        if len(tmp_instance['tokens']) < MAX_LENGTH:
            pretrain_paired_data.append(tmp_instance)
            TMPCOUNT += 1
        if TMPCOUNT >= NUMBERINSTANCEPERTYPE:
            break
    selected_glosses.append(tmp_gloss)
print('Number of kept training examples:', len(pretrain_paired_data))
selected_glosses.append('NA')
print('Number of glosses:', len(selected_glosses))

with open('data/pre_trained_glosses.json', 'w') as f:
    json.dump(selected_glosses, f)
with open('data/pre_trained_pairs.json', 'w') as f:
    json.dump(pretrain_paired_data, f)

# We then select the relevant data for the MAVEN dataset

maven_glosses = list()
for tmp_e_type in maven_e_type_to_definition:
    maven_glosses.append(maven_e_type_to_definition[tmp_e_type])

# We first load a sentence representation model to encode all candidate glosses
MPNet_encoder = SentenceTransformer('all-mpnet-base-v2')
device = torch.device("cuda")
MPNet_encoder.to(device)

candidate_gloss_embeddings = MPNet_encoder.encode(selected_glosses, batch_size=128, show_progress_bar=True,
                                                  convert_to_numpy=True)
maven_embeddings = MPNet_encoder.encode(maven_glosses, batch_size=128, show_progress_bar=True,
                                        convert_to_numpy=True)
np_searcher = NumpySearcher(candidate_gloss_embeddings)

neighbors, distances = np_searcher.search_batch(maven_embeddings.astype('float32'), 1)

warm_glosses = list()
for tmp_neighbor in neighbors:
    warm_glosses.append(selected_glosses[tmp_neighbor[0]])

warm_paired_data = list()
print(len(warm_glosses))
for tmp_gloss in warm_glosses:
    TMPCOUNT = 0
    for tmp_instance in gloss2instance[tmp_gloss]:
        if len(tmp_instance['tokens']) < MAX_LENGTH:
            warm_paired_data.append(tmp_instance)
            TMPCOUNT += 1
        if TMPCOUNT >= NUMBERINSTANCEPERTYPE:
            break
warm_glosses.append('NA')
print(len(warm_paired_data))

with open('data/warm_glosses.json', 'w') as f:
    json.dump(warm_glosses, f)
with open('data/warm_pairs.json', 'w') as f:
    json.dump(warm_paired_data, f)

print('end')
