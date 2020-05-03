import numpy as np
import matplotlib.pyplot as plt


       

# https://www.tensorflow.org/alpha/tutorials/text/image_captioning
def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
    
    
def evaluate(image, image_reader, token2id_table, id2token_table):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)
    
    
    features = image_reader.get_features([image])

    dec_input = tf.expand_dims([token2id_table.lookup('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions, axis=0)

        if id2token_table.lookup(predicted_id) == '<end>':
            return result, attention_plot
        
        result.append(token2id_table.lookup(predicted_id))
        dec_input = tf.expand_dims([predicted_id], 0)
        
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot
    

    

# TODO: compare with the padding in load_vte_dataset in datasets.py
def decode(arr, id2token):
    sentence = ''
    if arr is None:
        return 'output none'
    if arr.shape[0] == 0:
        return 'empty sentence'
    for i in arr:
        sentence += ' ' + id2token[i]
    if sentence == '':
        return 'empty empty sentence'
    return sentence

# def decode(arr, id2token):
#     sentence = ''
#     for i in arr:
#         sentence += ' ' + id2token[i]
#     return sentence

def encode(sentence, token2id):
    arr = [token2id['<start>']]
    for token in sentence.split(' '):
        arr.append(token2id[token])
    arr.append(token2id['<start>'])
    return arr


def ilabel2itoken(id2label, token2id):
    dico = {}
    for i in id2label:
        label = id2label[i]
        j = token2id[label]
        dico[i] = j
    return dico