import csv
import json

import numpy as np

from itertools import islice
from utils import batch

from utils import pad_sequences

from collections import Counter
from embeddings import load_glove


def load_te_dataset(filename, token2id, label2id, mode="hypothesis"):
    print(mode)
    labels = []
    padded_premises = []
    padded_hypotheses = []
    padded_explanations = []
    original_premises = []
    original_hypotheses = []
    original_explanations = []
    
    with open(filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        next(reader, None) #skip header
        for row in reader:
            label = row[0].strip()
            premise_tokens = row[1].strip().split()
            hypothesis_tokens = row[2].strip().split()
            premise = row[4].strip()
            hypothesis = row[5].strip()
            labels.append(label2id[label])
            padded_premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
            padded_hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])
            original_premises.append(premise)
            original_hypotheses.append(hypothesis)
            
            if mode=="explanation":
                explanation = row[7].strip()
                explanation_tokens = row[8].strip().split()
                padded_explanations.append([token2id.get(token, token2id["#unk#"]) for token in explanation_tokens])
                original_explanations.append(explanation)
        
        labels = np.array(labels)        
        
        padded_premises = pad_sequences(padded_premises, padding="post", value=token2id["#pad#"], dtype=np.long)
        padded_hypotheses = pad_sequences(padded_hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
        
        if mode=="explanation":
            # ExplainThenPredict: we want to predict label from explanation
            padded_explanations = pad_sequences(padded_explanations, padding="post", value=token2id["#pad#"], dtype=np.long)
            return labels, padded_premises, padded_explanations, original_premises, original_explanations
                    
        else:
            # Ablation: we want to predict label from hypothesis only
            return labels, padded_premises, padded_hypotheses, original_premises, original_hypotheses
        


def load_vte_dataset(nli_dataset_filename, token2id, label2id, keep_neutrals=True):
    labels = []
    padded_premises = []
    padded_hypotheses = []
    image_names = []
    original_premises = []
    original_hypotheses = []

    with open(nli_dataset_filename) as in_file:
        
        reader = csv.reader(in_file, delimiter="\t")
        
        next(reader, None) #skip header

        for row in reader:
            label = row[0].strip()

            if keep_neutrals==False and label == 'neutral':
                continue
                
            premise_tokens = row[1].strip().split()
            hypothesis_tokens = row[2].strip().split()
            image = row[3].strip().split("#")[0]
            
            premise = row[4].strip()
            hypothesis = row[5].strip()
            labels.append(label2id[label])

            padded_premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
            padded_hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])
            image_names.append(image)
            original_premises.append(premise)
            original_hypotheses.append(hypothesis)

        padded_premises = pad_sequences(padded_premises, padding="post", value=token2id["#pad#"], dtype=np.long)
        padded_hypotheses = pad_sequences(padded_hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

    return labels, padded_premises, padded_hypotheses, image_names, original_premises, original_hypotheses


# Code based on Vu et al., Coling18 (https://github.com/claudiogreco)
# TODO: add <start> and <end>

def load_e_vsnli_dataset(nli_dataset_filename, token2id, label2id, buffer_size=None, padding_length=None, keep_neutrals=True):
    labels = []
    padded_explanations = []
    padded_premises = []
    padded_hypotheses = []
    image_names = []
    pairIDs = []
    original_explanations = []
    original_premises = []
    original_hypotheses = []
    
    with open(nli_dataset_filename) as in_file:
        
        reader = csv.reader(in_file, delimiter="\t")
        
        next(reader, None) #skip header

        for i, row in enumerate(reader):
            
            if buffer_size and i >= buffer_size:
                break
            label = row[0].strip()

            if keep_neutrals==False and label == 'neutral':
                continue
                
            premise_tokens = row[1].strip().split()
            hypothesis_tokens = row[2].strip().split()
            image = row[3].strip().split("#")[0]
            premise = row[4].strip()
            hypothesis = row[5].strip()
            pairID = row[6]
            explanation = row[7].strip()
            explanation_tokens = row[8].strip().split()
            
            #TODO: add <start> and </end>
            explanation_tokens = ['<start>'] + explanation_tokens + ['<end>']
            hypothesis_tokens = ['<start>'] + hypothesis_tokens + ['<end>']
            
            labels.append(label2id[label])
            padded_premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
            padded_hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])
            padded_explanations.append([token2id.get(token, token2id["#unk#"]) for token in explanation_tokens])
            image_names.append(image)
            pairIDs.append(pairID)
            original_premises.append(premise)
            original_hypotheses.append(hypothesis)
            original_explanations.append(explanation)
        
        max_length = max(len(pad_expl) for pad_expl in padded_explanations)
        
        if padding_length is None:
            padding_length = max_length
          
        padded_premises = pad_sequences(padded_premises, maxlen=padding_length, padding="post", value=token2id["#pad#"], dtype=np.long)
        padded_hypotheses = pad_sequences(padded_hypotheses, maxlen=padding_length, padding="post", value=token2id["#pad#"], dtype=np.long)
        padded_explanations = pad_sequences(padded_explanations, maxlen=padding_length, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)
        pairIDs = np.array(pairIDs)
        

    return labels, padded_explanations, padded_premises, padded_hypotheses, image_names, original_explanations, original_premises, original_hypotheses, max_length, pairIDs


def load_e_vsnli_dataset_and_glove(nli_dataset_filename, label2id, vectors_filename, max_vocab, embeddings_size, buffer_size=None, padding_length=None, min_threshold=0, keep_neutrals=True):
    labels = []
    padded_explanations = []
    padded_premises = []
    padded_hypotheses = []
    image_names = []
    pairIDs = []
    original_explanations = []
    original_premises = []
    original_hypotheses = []
    
    all_premise_tokens = []
    all_hypothesis_tokens = []
    all_explanation_tokens = []
    
    with open(nli_dataset_filename) as in_file:
        
        reader = csv.reader(in_file, delimiter="\t")
        
        next(reader, None) #skip header

        for i, row in enumerate(reader):
            
            if buffer_size and i >= buffer_size:
                break
            label = row[0].strip()
            
            if keep_neutrals==False and label == 'neutral':
                continue
            premise_tokens = row[1].strip().split()
            hypothesis_tokens = row[2].strip().split()
            image = row[3].strip().split("#")[0]
            premise = row[4].strip()
            hypothesis = row[5].strip()
            pairID = row[6]
            explanation = row[7].strip()
            explanation_tokens = row[8].strip().split()
            labels.append(label2id[label])
            
            #TODO: add <start> and </end>
            explanation_tokens = ['<start>'] + explanation_tokens + ['<end>']
            hypothesis_tokens = ['<start>'] + hypothesis_tokens + ['<end>']
            
            all_premise_tokens.append(premise_tokens)
            all_hypothesis_tokens.append(hypothesis_tokens)
            all_explanation_tokens.append(explanation_tokens)
            
            image_names.append(image)
            pairIDs.append(pairID)
            original_premises.append(premise)
            original_hypotheses.append(hypothesis)
            original_explanations.append(explanation)
            
        labels = np.array(labels)
        pairIDs = np.array(pairIDs)
        
        if min_threshold:
            word_freq = Counter(x for xs in all_explanation_tokens for x in set(xs))
            word_freq = {x : word_freq[x] for x in word_freq if word_freq[x] >= min_threshold}
        else:
            word_freq = None
        
        embeddings, token2id, id2token = load_glove(vectors_filename, max_vocab, embeddings_size, word_freq)
       
        padded_premises = [[token2id.get(token, token2id["#unk#"]) for token in premise_tokens] 
                               for premise_tokens in all_premise_tokens]
        padded_hypotheses = [[token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens]
                                for hypothesis_tokens in all_hypothesis_tokens]
        padded_explanations = [[token2id.get(token, token2id["#unk#"]) for token in explanation_tokens]
                                   for explanation_tokens in all_explanation_tokens]
        
        max_length = max(len(pad_expl) for pad_expl in padded_explanations)
        if padding_length is None:
            padding_length = max_length
          
        padded_premises = pad_sequences(padded_premises, maxlen=padding_length, padding="post", value=token2id["#pad#"], dtype=np.long)
        padded_hypotheses = pad_sequences(padded_hypotheses, maxlen=padding_length, padding="post", value=token2id["#pad#"], dtype=np.long)
        padded_explanations = pad_sequences(padded_explanations, maxlen=padding_length, padding="post", value=token2id["#pad#"], dtype=np.long)
        

    return labels, padded_explanations, padded_premises, padded_hypotheses, image_names, original_explanations, original_premises, original_hypotheses, max_length, embeddings, token2id, id2token, pairIDs


# Filter for rare words

# def load_e_vsnli_dataset(nli_dataset_filename, token2id, label2id, buffer_size=None, min_threshold=0):
#     labels = []
#     padded_explanations = []
#     padded_premises = []
#     padded_hypotheses = []
#     image_names = []
#     original_explanations = []
#     original_premises = []
#     original_hypotheses = []
    
#     all_premise_tokens = []
#     all_hypothesis_tokens = []
#     all_explanation_tokens = []
    
#     with open(nli_dataset_filename) as in_file:
        
#         reader = csv.reader(in_file, delimiter="\t")
        
#         next(reader, None) #skip header

#         for i, row in enumerate(reader):
            
#             if buffer_size and i >= buffer_size:
#                 break
#             label = row[0].strip()
#             premise_tokens = row[1].strip().split()
#             hypothesis_tokens = row[2].strip().split()
#             image = row[3].strip().split("#")[0]
#             premise = row[4].strip()
#             hypothesis = row[5].strip()
#             explanation = row[7].strip()
#             explanation_tokens = row[8].strip().split()
#             labels.append(label2id[label])
            
#             all_premise_tokens.append(premise_tokens)
#             all_hypothesis_tokens.append(hypothesis_tokens)
#             all_explanation_tokens.append(explanation_tokens)
            
#             image_names.append(image)
#             original_premises.append(premise)
#             original_hypotheses.append(hypothesis)
#             original_explanations.append(explanation)


#         freq = Counter(x for xs in all_explanation_tokens for x in set(xs))
#         freq = {x : freq[x] for x in freq if freq[x] >= min_threshold}
        
#         token2id = {x: token2id[x] for x in token2id if x in freq or x in {'#unk#', '#pad#', '<start>', '<end>'}}
            
#         padded_premises = [[token2id.get(token, token2id["#unk#"]) for token in premise_tokens] 
#                                for premise_tokens in all_premise_tokens]
#         padded_hypotheses = [[token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens]
#                                 for hypothesis_tokens in all_hypothesis_tokens]
#         padded_explanations = [[token2id.get(token, token2id["#unk#"]) for token in explanation_tokens]
#                                    for explanation_tokens in all_explanation_tokens]
        
#         padded_premises = pad_sequences(padded_premises, padding="post", value=token2id["#pad#"], dtype=np.long)
#         padded_hypotheses = pad_sequences(padded_hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
#         padded_explanations = pad_sequences(padded_explanations, padding="post", value=token2id["#pad#"], dtype=np.long)
#         labels = np.array(labels)
#         max_length = max(len(pad_expl) for pad_expl in padded_explanations)
        

#     return labels, padded_explanations, padded_premises, padded_hypotheses, image_names, original_explanations, original_premises, original_hypotheses, max_length, token2id



def load_2task_dataset(nli_dataset_filename, token2id, label2id, keep_neutrals=True):
    labels = []
    padded_premises = []
    padded_hypotheses = []
    image_names = []
    original_premises = []
    original_hypotheses = []
    e_labels = []
    c_labels = []
    with open(nli_dataset_filename) as in_file:
        
        reader = csv.reader(in_file, delimiter="\t")
        
        next(reader, None) #skip header

        for row in reader:
            label = row[0].strip()
            if keep_neutrals==False and label == 'neutral':
                continue
            
            premise_tokens = row[1].strip().split()
            try:
                hypothesis_tokens = row[2].strip().split()
            except:
                break
            image = row[3].strip().split("#")[0]
            premise = row[4].strip()
            hypothesis = row[5].strip()
            labels.append(label2id[label])
            

            if label == 'entailment':
                e_labels.append(1)
            else:
                e_labels.append(0)
            if label == 'contradiction':
                c_labels.append(1)
            else:
                c_labels.append(0)
            padded_premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
            padded_hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])
            image_names.append(image)
            original_premises.append(premise)
            original_hypotheses.append(hypothesis)

        padded_premises = pad_sequences(padded_premises, padding="post", value=token2id["#pad#"], dtype=np.long)
        padded_hypotheses = pad_sequences(padded_hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)
        e_labels = np.array(e_labels)
        c_labels = np.array(c_labels)

    return labels, padded_premises, padded_hypotheses, image_names, original_premises, original_hypotheses, e_labels, c_labels



class ImageReader:
    def __init__(self, img_names_filename, img_features_filename):
        self._img_names_filename = img_names_filename
        self._img_features_filename = img_features_filename

        with open(img_names_filename) as in_file:
            img_names = json.load(in_file)

        with open(img_features_filename, mode="rb") as in_file:
            img_features = np.load(in_file)

        self._img_names_features = {filename: features for filename, features in zip(img_names, img_features)}

    def get_features(self, images_names):
        return np.array([self._img_names_features[image_name] for image_name in images_names])
    
    
    
