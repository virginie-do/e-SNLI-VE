import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
from embeddings import glove_embeddings_initializer, load_glove

from utils import gated_tanh
from utils import pad_sequences

from utils_explain import decode, encode, beam_search_decoder


# Decoder to generate explanations
# RNN with attention over image + hypothesis
# From tensorflow image captioning tutorial
# Show, attend and tell

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights



class RNN_Decoder(tf.keras.Model):
    #def __init__(self, embedding_dim, units, vocab_size, vocab_proj_dim):
    def __init__(self, embedding_matrix, units, vocab_size, vocab_proj_dim):
        super(RNN_Decoder, self).__init__()
        self.units = units
        
        self.vocab_proj_dim = vocab_proj_dim

        #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.embedding = lambda x: tf.nn.embedding_lookup(embedding_matrix, x)
        
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        
        # Add projection
        if self.vocab_proj_dim:
            self.vocab_proj = tf.keras.layers.Dense(self.vocab_proj_dim)
        
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
        #self.attention = tf.contrib.seq2seq.BahdanauAttention(self.units)
     

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        
        #this is done before the decoder
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #x = tf.concat([tf.expand_dims(context_vector, 0), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        
        # Add vocab projection:
        # x shape = (batch_size * max_length, vocab_proj_dim)
        if self.vocab_proj_dim:
            x = self.vocab_proj(x)
        
        # fc2 is vocab_layer
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights


    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
    

class eVSNLI_net(tf.keras.Model):
    def __init__(self,
                 num_tokens,
                 embeddings,
                 embeddings_size,
                 train_embeddings,
                 dropout_input,
                 rnn_hidden_size,
                 id2token,
                 token2id,
                 id2label,
                 label2id,
                 mode='teacher',
                 vocab_proj_dim=None):
        super(eVSNLI_net, self).__init__()
        
        self.mode = mode
        assert mode == 'teacher' or 'forloop'
        
        self.num_tokens = num_tokens
        
        self.lstm_cell = DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(rnn_hidden_size),
            input_keep_prob=dropout_input,
            output_keep_prob=dropout_input
        )
        
        if embeddings is not None:
            self.embedding_matrix = tf.get_variable(
                "embedding_matrix",
                shape=(num_tokens, embeddings_size),
                initializer=glove_embeddings_initializer(embeddings),
                trainable=train_embeddings
            )
            print("Loaded GloVe embeddings!")
        else:
            self.embedding_matrix = tf.get_variable(
                "embedding_matrix",
                shape=(num_tokens, embeddings_size),
                initializer=tf.random_normal_initializer(stddev=0.05),
                trainable=train_embeddings
            )
        
        #vocab_proj_dim for vocab projection
        #self.decoder = RNN_Decoder(embeddings_size, rnn_hidden_size, num_tokens, vocab_proj_dim)
        self.decoder = RNN_Decoder(self.embedding_matrix, rnn_hidden_size, num_tokens, vocab_proj_dim)
              
        keys = list(token2id.keys())
        values = [token2id[k] for k in keys]
        self.token2id_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int64),-1)

        mapping_token = tf.constant(list(id2token.values()), dtype=tf.string)
        self.id2token_table = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping_token,
            default_value="#unk#",
            name=None
        )

        mapping_label = tf.constant(list(id2label.values()), dtype=tf.string)
        self.id2label_table = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping_label,
            default_value="#unk#",
            name=None
        )

    
    def call(self, 
            premise_input,
            hypothesis_input,
            img_features_input,
            label_input,
            target_expl,
            target_length,
            dropout_input,
            num_labels,
            num_img_features,
            img_features_size,
            rnn_hidden_size,
            multimodal_fusion_hidden_size,
            classification_hidden_size,
            max_length):
        
        hypothesis_length = tf.cast(
            tf.reduce_sum(
                tf.cast(tf.not_equal(hypothesis_input, tf.zeros_like(hypothesis_input, dtype=tf.int32)), tf.int64),
                1
            ),
            tf.int32
        )

        hypothesis_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, hypothesis_input)

        hypothesis_outputs, hypothesis_final_states = tf.nn.dynamic_rnn(
            cell=self.lstm_cell,
            inputs=hypothesis_embeddings,
            sequence_length=hypothesis_length,
            dtype=tf.float32
        )
        normalized_img_features = tf.nn.l2_normalize(img_features_input, dim=2)

        reshaped_hypothesis = tf.reshape(tf.tile(hypothesis_final_states.h, [1, num_img_features]),
                                         [-1, num_img_features, rnn_hidden_size])
        img_hypothesis_concatenation = tf.concat([normalized_img_features, reshaped_hypothesis], -1)
        gated_img_hypothesis_concatenation = tf.nn.dropout(
            gated_tanh(img_hypothesis_concatenation, rnn_hidden_size),
            keep_prob=dropout_input
        )
        att_wa_hypothesis = lambda x: tf.nn.dropout(
            tf.contrib.layers.fully_connected(x, 1, activation_fn=None, biases_initializer=None),
            keep_prob=dropout_input
        )
        a_hypothesis = att_wa_hypothesis(gated_img_hypothesis_concatenation)
        a_hypothesis = tf.nn.softmax(tf.squeeze(a_hypothesis))
        v_head_hypothesis = tf.squeeze(tf.matmul(tf.expand_dims(a_hypothesis, 1), normalized_img_features))

        gated_hypothesis = tf.nn.dropout(
            gated_tanh(hypothesis_final_states.h, multimodal_fusion_hidden_size),
            keep_prob=dropout_input
        )
        v_head_hypothesis.set_shape((hypothesis_embeddings.get_shape()[0], img_features_size))
           
        gated_img_features_hypothesis = tf.nn.dropout(
            gated_tanh(v_head_hypothesis, multimodal_fusion_hidden_size),
            keep_prob=dropout_input
        )
        h_hypothesis_img = tf.multiply(gated_hypothesis, gated_img_features_hypothesis)

        # Features used to classify label and generate explanation
        final_concatenation = tf.concat([h_hypothesis_img], 1)

        # Classifier
        gated_first_layer = tf.nn.dropout(
            gated_tanh(final_concatenation, classification_hidden_size),
            keep_prob=dropout_input
        )
        gated_second_layer = tf.nn.dropout(
            gated_tanh(gated_first_layer, classification_hidden_size),
            keep_prob=dropout_input
        )
        gated_third_layer = tf.nn.dropout(
            gated_tanh(gated_second_layer, classification_hidden_size),
            keep_prob=dropout_input
        )

        pred_label = tf.contrib.layers.fully_connected(
            gated_third_layer,
            num_labels,
            activation_fn=None
        )

        # insert GRU here to generate explanations
        # expl= (bs, T, 300)
        
        start_token = tf.constant('<start>', dtype=tf.string)
        end_token = tf.constant('<end>', dtype=tf.string)
        
        batch_size = tf.shape(hypothesis_input)[0]

        #if tf.reduce_all(tf.math.equal(mode, tf.constant('teacher', dtype=tf.string))):
        # teacher forcing
        if self.mode == 'teacher':
            
            print("teacher")

            hidden_t = self.decoder.reset_state(batch_size = batch_size)

            batch_start_token = tf.fill([batch_size], '<start>')
            batch_end_token = tf.fill([batch_size], '<end>')
            dec_input_t = tf.expand_dims(self.token2id_table.lookup(batch_start_token), 1)

            all_predictions = []
            
            # TODO: why target_expl.shape[1] gives None?
            # replacing with max_length but bad
            
            #for t in range(1, tf.shape(target_expl)[1])
            #for t in tf.range(self.explanation_length_input):
            for t in range(1, max_length+1):
                # passing the features through the decoder
                predictions, hidden_t, attention_weights = self.decoder(dec_input_t, tf.expand_dims(final_concatenation, 1), hidden_t)
                
                #prepend label
                if t == 1 and label_input is not None:
                    labels = self.id2label_table.lookup(label_input) #(bs,)
                    dec_input_t = self.token2id_table.lookup(labels) #(bs,)
                    dec_input_t = tf.expand_dims(dec_input_t, 1) #(bs,1)
                    
                # using teacher forcing
                #if t < max_length:
                elif t < max_length:
                    dec_input_t = tf.expand_dims(target_expl[:, t], 1)
                else:
                    dec_input_t = tf.expand_dims(self.token2id_table.lookup(batch_end_token), 1)                       
                    
                # predictions: (bs, 1, n_vocab)
                all_predictions.append(predictions)
                
            # all_predictions: (bs, T, n_vocab)
            all_predictions = tf.stack(all_predictions, axis=1)

            return pred_label, all_predictions


        else:
            print("forloop")
            #all_predictions = []
            
            ##TODO: attention shape
            #attention_features_shape = 36
            #all_attention_plots = []
            
            # pred_expls is a list of strings of size batch_size
            #pred_expls = [""] * batch_size
            #finished = [False] * batch_size
            
            #pred_expls = tf.fill([batch_size], "")
            #finished = tf.fill([batch_size], False)
            
            #TODO
            pred_expls = []
            pred_expls_words = []
            #finished = tf.zeros((batch_size))
            
            t = 0
            hidden_t = self.decoder.reset_state(batch_size = batch_size)

            batch_start_token = tf.fill([batch_size], '<start>')
            dec_input_t = tf.expand_dims(self.token2id_table.lookup(batch_start_token), 1)
            
            #TODO:
            #while t < max_length and tf.reduce_sum(finished) != batch_size:
            while t < max_length: 
                t += 1

                #dec_output_t: (bs, max_vocab)
                #dec_output_t: (bs * max_length, max_vocab)
                dec_output_t, hidden_t, attention_weights = self.decoder(dec_input_t, tf.expand_dims(final_concatenation, 1), hidden_t)
                
                #predicted_id: (bs* max_length) or (bs*max_length, 1)
                predicted_id = tf.argmax(dec_output_t, axis=1)
                pred_expls.append(predicted_id)
                pred_expls_words.append(self.id2token_table.lookup(predicted_id))
                
                # TODO
                #completed = tf.where(predicted_id == self.token2id_table.lookup(end_token))
                #finished[completed] = 1
                
                if t > 1:
                #if True:
                    #dec_input_t = tf.expand_dims(predicted_id, 1)
                    dec_input_t = tf.reshape(predicted_id, [batch_size, 1])

                else:
                    out_labels = tf.argmax(pred_label, axis=1)
                    # pred_label IDs --> labels words --> embeddings
                    labels = self.id2label_table.lookup(out_labels) #(bs,)
                    dec_input_t = self.token2id_table.lookup(labels) #(bs,)
                    dec_input_t = tf.expand_dims(dec_input_t, 1) #(bs,1)
                    
                #all_predictions.append(dec_output_t)
                
            #all_predictions = tf.stack(all_predictions, axis=1)
            pred_expls = tf.stack(pred_expls, axis=1)
            pred_expls_words = tf.stack(pred_expls_words, axis=1)
            return pred_label, pred_expls_words
            #return pred_label, all_predictions#, a_hypothesis
