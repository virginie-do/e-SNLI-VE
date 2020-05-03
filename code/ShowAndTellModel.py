
"""Builds the model.

Inputs:
  image_feature
  input_seqs
  keep_prob 
  target_seqs 
  input_mask 
Outputs:
  total_loss 
  preds 
"""

import tensorflow as tf
from embeddings import glove_embeddings_initializer
from utils import gated_tanh
from utils import pad_sequences
import random
import numpy as np

def build_model(config, embeddings, mode, ilabel2itoken=None, inference_batch = None):

    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train" or "inference".
      inference_batch: if mode is 'inference', we will need to provide the batch_size of input data. Otherwise, leave it as None. 
      glove_vocab: if we need to use glove word2vec to initialize our vocab embeddings, we will provide with a matrix of [config.vocab_size, config.embedding_size]. If not, we leave it as None. 
    """
    assert mode in ["train", "inference"]
    if mode == 'inference' and inference_batch is None:
        raise ValueError("When inference mode, inference_batch must be provided!")
    config = config

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    initializer = tf.random_uniform_initializer(
        minval=-config.initializer_scale,
        maxval=config.initializer_scale)
    
    ### Inputs for VQA model ###

    hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
    img_features_input = tf.placeholder(tf.float32, (None, config.num_img_features, config.img_features_size),
                                        name="img_features_input")
    label_input = tf.placeholder(tf.int32, (None,), name="label_input")
    dropout_input = tf.placeholder(tf.float32, name="dropout_input")
    
    ### Inputs for explanation generation ###

    # An int32 Tensor with shape [batch_size, padded_length].
    input_seqs = tf.placeholder(tf.int32, [None, None], name='input_seqs')

    # An int32 Tensor with shape [batch_size, padded_length].
    target_seqs = tf.placeholder(tf.int32, [None, None], name='target_seqs')    
    
    # A float32 Tensor with shape [1]
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
    
    # A float32 Tensor with shape [batch_size, image_feature_size].
    image_feature = tf.placeholder(tf.float32, [None, config.image_feature_size], name='image_feature')

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    seq_embedding = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    inception_variables = []

    # Global step Tensor.
    global_step = None
    
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
    # Dynamic batch size
    batch_size = tf.shape(hypothesis_input)[0]
    
    # Table to map label_id to token_id
    if ilabel2itoken:
        keys = list(ilabel2itoken.keys())
        values = [ilabel2itoken[k] for k in keys]
        ilabel2itoken_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int32, value_dtype=tf.int32),-1)
    
    ### Builds the input sequence embeddings ###
    # Inputs:
    #   self.input_seqs
    # Outputs:
    #   self.seq_embeddings
    ############################################

#     with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
#         if glove_vocab is None:
#             embedding_map = tf.get_variable(
#                 name="map",
#                 shape=[config.vocab_size, config.embedding_size],
#                 initializer=initializer)
#         else:
#             init = tf.constant(glove_vocab.astype('float32'))
#             embedding_map = tf.get_variable(
#                 name="map",
#                 initializer=init)
#         seq_embedding = tf.nn.embedding_lookup(embedding_map, input_seqs)
        
    with tf.variable_scope("hypothesis_embeddings"), tf.device("/cpu:0"):    
        if embeddings is not None:
            embedding_map = tf.get_variable(
                "map",
                shape=[config.vocab_size, config.embedding_size],
                initializer=glove_embeddings_initializer(embeddings),
                trainable=config.train_embeddings
            )
            print("Loaded GloVe embeddings!")
        else:
            embedding_map = tf.get_variable(
                "map",
                shape=[config.vocab_size, config.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.05),
                trainable=config.train_embeddings #TODO
            )
        hypothesis_embeddings = tf.nn.embedding_lookup(embedding_map, hypothesis_input)
        

    ############ Builds the model ##############
    # Inputs:
    #   self.image_feature
    #   self.seq_embeddings
    #   self.target_seqs (training and eval only)
    #   self.input_mask (training and eval only)
    # Outputs:
    #   self.total_loss (training and eval only)
    #   self.target_cross_entropy_losses (training and eval only)
    #   self.target_cross_entropy_loss_weights (training and eval only)
    ############################################
    
    ############ VQA part ######################

    hypothesis_length = tf.cast(
        tf.reduce_sum(
            tf.cast(tf.not_equal(hypothesis_input, tf.zeros_like(hypothesis_input, dtype=tf.int32)), tf.int64),
            1
        ),
        tf.int32
    )
        
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.LSTMCell(config.num_lstm_units),
        input_keep_prob=dropout_input,
        output_keep_prob=dropout_input
    )
    hypothesis_outputs, hypothesis_final_states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=hypothesis_embeddings,
        sequence_length=hypothesis_length,
        dtype=tf.float32
    )
    normalized_img_features = tf.nn.l2_normalize(img_features_input, dim=2)

    reshaped_hypothesis = tf.reshape(tf.tile(hypothesis_final_states.h, [1, config.num_img_features]),
                                     [-1, config.num_img_features, config.num_lstm_units])
    img_hypothesis_concatenation = tf.concat([normalized_img_features, reshaped_hypothesis], -1)
    gated_img_hypothesis_concatenation = tf.nn.dropout(
        gated_tanh(img_hypothesis_concatenation, config.num_lstm_units),
        keep_prob=dropout_input
    )
    att_wa_hypothesis = lambda x: tf.nn.dropout(
        tf.contrib.layers.fully_connected(x, 1, activation_fn=None, biases_initializer=None),
        keep_prob=dropout_input
    )
    a_hypothesis = att_wa_hypothesis(gated_img_hypothesis_concatenation)
    a_hypothesis = tf.nn.softmax(tf.squeeze(a_hypothesis, axis=-1))
    
    v_head_hypothesis = tf.squeeze(tf.matmul(tf.expand_dims(a_hypothesis, 1), normalized_img_features), axis=1)

    gated_hypothesis = tf.nn.dropout(
        gated_tanh(hypothesis_final_states.h, config.multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )
    v_head_hypothesis.set_shape((hypothesis_embeddings.get_shape()[0], config.img_features_size))
    gated_img_features_hypothesis = tf.nn.dropout(
        gated_tanh(v_head_hypothesis, config.multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )
    h_hypothesis_img = tf.multiply(gated_hypothesis, gated_img_features_hypothesis)

    final_concatenation = tf.concat([h_hypothesis_img], 1)
    gated_first_layer = tf.nn.dropout(
        gated_tanh(final_concatenation, config.classification_hidden_size),
        keep_prob=dropout_input
    )
    gated_second_layer = tf.nn.dropout(
        gated_tanh(gated_first_layer, config.classification_hidden_size),
        keep_prob=dropout_input
    )
    gated_third_layer = tf.nn.dropout(
        gated_tanh(gated_second_layer, config.classification_hidden_size),
        keep_prob=dropout_input
    )

    label_logits = tf.contrib.layers.fully_connected(
        gated_third_layer,
        config.num_labels,
        activation_fn=None
    )
     
    ############## Explanation generation part ######################
    multimodal_feature = final_concatenation
    
    if mode == 'train' and ilabel2itoken:
        # prepend gold label
        # done outside of the build function in inference mode
        pre_labels = ilabel2itoken_table.lookup(label_input)
        input_seqs = tf.concat([tf.expand_dims(pre_labels, 1), input_seqs], axis=1)

    
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        seq_embedding = tf.nn.embedding_lookup(embedding_map, input_seqs)
        
    
    lstm_cell_expl = tf.nn.rnn_cell.LSTMCell(
        num_units=config.num_lstm_units, state_is_tuple=True)
        
    lstm_cell_expl = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell_expl,
        input_keep_prob=keep_prob,
        output_keep_prob=keep_prob)

    # TODO: attention?
    #attn_meca = tf.contrib.seq2seq.BahdanauAttention(config.num_lstm_units, multimodal_feature)   
    #attn_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_cell_expl, attn_meca, output_attention=False)
    
    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:

        # Feed the image embeddings to set the initial LSTM state.
        if mode == 'train':
            zero_state = lstm_cell_expl.zero_state(
                batch_size=batch_size, dtype=tf.float32)
            #zero_state = attn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            
        elif mode == 'inference':
            zero_state = lstm_cell_expl.zero_state(
                batch_size=inference_batch, dtype=tf.float32)
            #zero_state = attn_cell.zero_state(batch_size=inference_batch, dtype=tf.float32)

        with tf.variable_scope('multimodal_embeddings'):
            multimodal_embeddings = tf.contrib.layers.fully_connected(
                  inputs=multimodal_feature,
                  num_outputs=config.embedding_size,
                  activation_fn=None,
                  weights_initializer=initializer,
                  biases_initializer=None)

        _, initial_state = lstm_cell_expl(multimodal_embeddings, zero_state)
        #_, initial_state = attn_cell(multimodal_embeddings, zero_state)
        

        # Allow the LSTM variables to be reused.
        lstm_scope.reuse_variables()

        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(input_mask, 1)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell_expl,
                                                    inputs=seq_embedding,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)
        
#         lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=attn_cell,
#                                                     inputs=seq_embedding,
#                                                     sequence_length=sequence_length,
#                                                     initial_state=initial_state,
#                                                     dtype=tf.float32,
#                                                     scope=lstm_scope)

        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell_expl.output_size]) # output_size == 256
        #lstm_outputs = tf.reshape(lstm_outputs, [-1, attn_cell.output_size]) # output_size == 256
      
    with tf.variable_scope('logits'):
        W = tf.get_variable('W', [lstm_cell_expl.output_size, config.vocab_size], initializer=initializer)
        #W = tf.get_variable('W', [attn_cell.output_size, config.vocab_size], initializer=initializer)
        b = tf.get_variable('b', [config.vocab_size], initializer=tf.constant_initializer(0.0))
        
        logits = tf.matmul(lstm_outputs, W) + b # logits: [batch_size * padded_length, config.vocab_size]
          
    ###### for inference & validation only #######
    softmax = tf.nn.softmax(logits)
    preds = tf.argmax(softmax, 1)
    ##############################################
    
    # for training only below 
    targets = tf.reshape(target_seqs, [-1])
    weights = tf.to_float(tf.reshape(input_mask, [-1]))

    # Compute losses.
    
    label_loss = tf.losses.sparse_softmax_cross_entropy(label_input, label_logits)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
    
    explanation_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                        tf.reduce_sum(weights), name="explanation_loss") 
    batch_loss = (1-config.alpha) * explanation_loss + config.alpha * label_loss
    tf.contrib.losses.add_loss(batch_loss)
    total_loss = tf.contrib.losses.get_total_loss()
    
    # target_cross_entropy_losses = losses  # Used in evaluation.
    # target_cross_entropy_loss_weights = weights  # Used in evaluation.
    
    
    # TODO; what else should I return?

    return dict(
        total_loss = total_loss, 
        global_step = global_step, 
        image_feature = image_feature, 
        input_mask = input_mask, 
        target_seqs = target_seqs, 
        input_seqs = input_seqs, 
        final_state = final_state,
        initial_state = initial_state, 
        softmax = softmax,
        preds = preds, 
        keep_prob = keep_prob, 
        saver = tf.train.Saver(),
        hypothesis_input = hypothesis_input,
        img_features_input = img_features_input,
        label_input = label_input,
        dropout_input = dropout_input,
        label_logits = label_logits,
        explanation_loss = explanation_loss,
        attention_output = a_hypothesis
    )

