import numpy as np
import tensorflow as tf


def load_glove(filename, max_vocab, embedding_size, freq=None):
    #embeddings = np.zeros((max_vocab + 2, embedding_size), dtype=np.float32)
    embeddings = []
    token2id = {}
    id2token = {}

    token_id = len(token2id)
    token2id["#pad#"] = token_id
    id2token[token_id] = "#pad#"
    embeddings.append(np.zeros(embedding_size, dtype=np.float32))
 
    token_id = len(token2id)
    token2id["#unk#"] = token_id
    id2token[token_id] = "#unk#"
    embeddings.append(np.zeros(embedding_size, dtype=np.float32))
    
    token_id = len(token2id)  
    token2id["<start>"] = token_id
    id2token[token_id] = "<start>"
    embeddings.append(np.zeros(embedding_size, dtype=np.float32))
    
    token_id = len(token2id)  
    token2id["<end>"] = token_id
    id2token[token_id] = "<end>"
    embeddings.append(np.zeros(embedding_size, dtype=np.float32))

    special_words = {'entailment', 'neutral', 'contradiction', '#pad#', '#unk#', '<start>', '<end>'}

    with open(filename) as in_file:
        for line_index, line in enumerate(in_file):
            values = line.rstrip().split(" ")
            word = values[0]
            if freq and word not in freq.keys() | special_words:
                #print(freq)
                continue
            embedding = np.array(values[1:], dtype=np.float32)
            token_id = len(token2id)
            token2id[word] = token_id
            id2token[token_id] = word
            embeddings.append(embedding)

            if token_id == max_vocab + 1:
                break
                
    embeddings = np.array(embeddings)
    #assert embeddings.shape[0] == len(token2id)
    return embeddings, token2id, id2token



def glove_embeddings_initializer(embeddings):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return embeddings

    return _initializer
