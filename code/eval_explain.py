from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys

from datetime import datetime 
import configuration
from ShowAndTellModel import build_model

import numpy as np
import scipy.misc

from explanation_generator import * 

FLAGS = None
model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

verbose = True
mode = 'inference'

###

import atexit
import csv
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from datasets import ImageReader, load_e_vsnli_dataset

from utils import batch
from utils import start_logger, stop_logger

from utils_explain import decode, encode
from nltk.translate.bleu_score import corpus_bleu
###


def build_parser():
    
    parser = ArgumentParser()
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--model_filename", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    parser.add_argument("--result_filename", type=str, required=True)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--imbalance", type=bool, default=False)
    return parser



def run_inference(sess, hypotheses, img_features, generator, keep_prob):

    batch_size = img_features.shape[0]
    final_pred_expls = []

    for i in range(batch_size):
        hypothesis = np.expand_dims(hypotheses[i], axis=0)
        img_feature = np.expand_dims(img_features[i], axis=0)
        pred_expl = generator.beam_search(sess, hypothesis, img_feature)
        pred_expl = pred_expl[0].sentence
        
        final_pred_expls.append(np.array(pred_expl))
        
    return final_pred_expls


def run_inference_attn(sess, hypotheses, img_features, generator, keep_prob):

    batch_size = img_features.shape[0]
    final_pred_expls = []
    final_pred_attns = []

    for i in range(batch_size):
        hypothesis = np.expand_dims(hypotheses[i], axis=0)
        img_feature = np.expand_dims(img_features[i], axis=0)
        pred_attn, pred_expl = generator.beam_search(sess, hypothesis, img_feature)
        pred_expl = pred_expl[0].sentence
        
        final_pred_expls.append(np.array(pred_expl))
        final_pred_attns.append(np.array(pred_attn))
        
    return final_pred_attns, final_pred_expls



def main(_):
    
    BATCH_SIZE_INFERENCE = 1
    
    random_seed = 12345
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    
    start_logger(FLAGS.result_filename + ".log")
    atexit.register(stop_logger)

    print("-- Loading params")
    with open(FLAGS.model_filename + ".params", mode="r") as in_file:
        params = json.load(in_file)

    print("-- Loading index")
    with open(FLAGS.model_filename + ".index", mode="rb") as in_file:
        index = pickle.load(in_file)
        token2id = index["token2id"]
        id2token = index["id2token"]
        label2id = index["label2id"]
        id2label = index["id2label"]
        num_tokens = len(token2id)
        num_labels = len(label2id)
        
    print("Number of tokens: {}".format(num_tokens))
    print("Number of labels: {}".format(num_labels))
    
    model_config.set_vocab_size(num_tokens)
    print("Vocab size set!")
    
    print("-- Loading test set")
    test_labels, test_padded_explanations, test_padded_premises, test_padded_hypotheses, test_img_names, test_original_explanations, test_original_premises, test_original_hypotheses, test_max_length, test_pairIDs = \
        load_e_vsnli_dataset(
            FLAGS.test_filename,
            token2id,
            label2id,
            buffer_size=FLAGS.buffer_size,
        )
    
    if FLAGS.imbalance == True:
        #class_freqs = np.load(FLAGS.model_filename + '_class_freqs.npy')
        
        test_num_examples = test_labels.shape[0]
        class_freqs = np.bincount(test_labels) / test_num_examples
        class_weights = 1 / (class_freqs * num_labels)
        print("Class frequencies: ", class_freqs)
        print("Weights: ", class_weights)
        
    test_original_premises = np.array(test_original_premises)
    test_original_hypotheses = np.array(test_original_hypotheses)
    test_original_explanations = np.array(test_original_explanations)
    
    print("-- Loading images")
    image_reader = ImageReader(FLAGS.img_names_filename, FLAGS.img_features_filename)
    
    
    model_config.set_vocab_size(num_tokens)
    model_config.set_alpha(params['alpha'])
     
    # Build the TensorFlow graph and train it
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        
        model = build_model(model_config, embeddings=None, mode=mode, inference_batch=BATCH_SIZE_INFERENCE)
        
        generator = AttentionExplanationGenerator(model, 
                                         vocab=token2id, 
                                         max_explanation_length=model_config.padded_length-1)
    
        # run training 
        init = tf.global_variables_initializer()
        with tf.Session() as session:

            session.run(init)

            model['saver'].restore(session, FLAGS.model_filename + ".ckpt")

              
            print("Model restored! Last step run: ", session.run(model['global_step']))

            print("-- Evaluating model")
            test_num_examples = test_labels.shape[0]
            test_batches_indexes = np.arange(test_num_examples)
            test_num_correct = 0
            y_true = []
            y_pred = []     


            with open(FLAGS.result_filename + ".predictions", mode="w") as out_file:
                writer = csv.writer(out_file, delimiter="\t")
                for indexes in batch(test_batches_indexes, FLAGS.batch_size):
                    
                    test_batch_pairIDs = test_pairIDs[indexes]
                    
                    test_batch_premises = test_padded_premises[indexes]
                    test_batch_hypotheses = test_padded_hypotheses[indexes]
                    test_batch_labels = test_labels[indexes]
                    test_batch_explanations = test_padded_explanations[indexes]
                    batch_img_names = [test_img_names[i] for i in indexes]
                    batch_img_features = image_reader.get_features(batch_img_names)

                    test_batch_original_premises = test_original_premises[indexes]
                    test_batch_original_hypotheses = test_original_hypotheses[indexes]
                    test_batch_original_explanations = test_original_explanations[indexes]
  
                    pred_attns, pred_explanations = run_inference_attn(session, test_batch_hypotheses, batch_img_features, generator, 1.0)
                    
                    # don't decode the first token which corresponds to the prepended label
                    # nor the last because it is <end>
                    pred_explanations_decoded = [decode(pred_explanations[i][1:-1], id2token) for i in range(len(indexes))]
                    
                    #batch_bleu = corpus_bleu(test_batch_original_explanations, pred_explanations_decoded)
                    #print("Current BLEU score: ", batch_bleu)

                    # add explanations in result file 
                    for i in range(len(indexes)):

                        writer.writerow(
                            [
                                id2label[test_batch_labels[i]],
                                " ".join([id2token[id] for id in test_batch_premises[i] if id != token2id["#pad#"]]),
                                " ".join([id2token[id] for id in test_batch_hypotheses[i] if id != token2id["#pad#"]]),
                                batch_img_names[i],
                                test_batch_original_premises[i],
                                test_batch_original_hypotheses[i],
                                " ".join([id2token[id] for id in test_batch_explanations[i] if id != token2id["#pad#"]]),
                                pred_explanations_decoded[i],
                                list(np.where(pred_attns[i]>0.05)[0]),
                                #pred_attns[i][0],
                                test_batch_pairIDs[i]
                            ]
                        )

            data = pd.read_csv(
                FLAGS.result_filename + ".predictions",
                sep="\t",
                header=None,
                names=["gold_label", "premise_toks", "hypothesis_toks", "jpg", "premise", "hypothesis", "original_explanation", "generated_explanation", "top_rois"]
            )


            
if __name__ == '__main__':
    parser = build_parser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)