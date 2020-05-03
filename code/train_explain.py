"""Train the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datetime import datetime 
import configuration
from ShowAndTellModel import build_model
import numpy as np
import os
import sys

####

import atexit
import json

import pickle
import random
from argparse import ArgumentParser


from datasets import ImageReader, load_e_vsnli_dataset, load_e_vsnli_dataset_and_glove
from embeddings import glove_embeddings_initializer, load_glove

from utils import Progbar
from utils import batch
from utils import start_logger, stop_logger

from utils import pad_sequences
from utils_explain import decode, ilabel2itoken


def build_parser():
    
    parser = ArgumentParser()
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--dev_filename", type=str, required=True)
    parser.add_argument("--vectors_filename", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    parser.add_argument("--model_save_filename", type=str, required=True)
    
    parser.add_argument("--max_vocab", type=int, default=300002)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dev_batch_size", type=int, default=32)

    parser.add_argument("--l2_reg", type=float, default=0.000005)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0)

    parser.add_argument("--min_threshold", type=int, default=None)
    
    parser.add_argument("--imbalance", type=bool, default=False)

    parser.add_argument(
      '--print_every',
      type=int,
      default=100,
      help='Num of steps to print your training loss. 0 for not printing/'
    )
    parser.add_argument(
      '--sample_every',
      type=int,
      default=50,
      help='Num of steps to generate captions on some validation images. 0 for not validating.'
    )
    
    return parser


model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

FLAGS = None 
mode = 'train'

# TODO: perplexity and teacher forcing for ExplainThenPredict
def _run_validation(sess, dev_batch_hypotheses, dev_batch_labels, dev_batch_explanations, dev_batch_img_features, batch_size, ilabel2itoken, model, keep_prob):
    """
    Here we get the loss but don't update the gradient 
    """
    explanations_in = dev_batch_explanations[:, :-1]
    explanations_out = dev_batch_explanations[:, 1:]

    mask = (explanations_out != model_config._null)

    explanation_loss_value = sess.run([model['explanation_loss']], 
                                  feed_dict={model['hypothesis_input']: dev_batch_hypotheses, 
                                             model['label_input']: dev_batch_labels, 
                                             model['img_features_input']: dev_batch_img_features, 
                                             model['input_seqs']: explanations_in, 
                                             model['target_seqs']: explanations_out,
                                             model['dropout_input']: 1.0,
                                             model['input_mask']: mask, 
                                             model['keep_prob']: keep_prob})
    #perplexity = np.exp(expl_loss)
    #TODO: add it up and exp() to get perplexity
    return explanation_loss_value



def _step(sess, batch_hypotheses, batch_labels, batch_explanations, batch_img_features, train_op, model, keep_prob):
    """
    Make a single gradient update for batch data. 
    """
    
    explanations_in = batch_explanations[:, :-1]
    explanations_out = batch_explanations[:, 1:]

    mask = (explanations_out != model_config._null)

    _, total_loss_value= sess.run([train_op, model['total_loss']], 
                                  feed_dict={model['hypothesis_input']: batch_hypotheses, 
                                             model['label_input']: batch_labels, 
                                             model['img_features_input']: batch_img_features, 
                                             model['input_seqs']: explanations_in, 
                                             model['target_seqs']: explanations_out,
                                             model['dropout_input']: 1.0,
                                             model['input_mask']: mask, 
                                             model['keep_prob']: keep_prob})

    return total_loss_value


def main(_):
    
    random_seed = 12345
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    start_logger(FLAGS.model_save_filename + ".train_log")
    atexit.register(stop_logger)

    print("-- Building vocabulary")
    #embeddings, token2id, id2token = load_glove(args.vectors_filename, args.max_vocab, args.embeddings_size)
    
    label2id = {"neutral": 0, "entailment": 1, "contradiction": 2}
    id2label = {v: k for k, v in label2id.items()}
    
    #num_tokens = len(token2id)
    
    num_labels = len(label2id)
    
    #print("Number of tokens: {}".format(num_tokens))
    print("Number of labels: {}".format(num_labels))
    
    
    # Load e_vsnli 
    # Explanations are encoded/padded, we ignore original explanations
    print("-- Loading training set")

    train_labels, train_explanations, train_premises, train_hypotheses, train_img_names, _, _, _, train_max_length, embeddings, token2id, id2token, _ = \
        load_e_vsnli_dataset_and_glove(
            FLAGS.train_filename,
            label2id,
            FLAGS.vectors_filename, 
            FLAGS.max_vocab, 
            model_config.embedding_size,
            buffer_size=FLAGS.buffer_size,
            min_threshold = FLAGS.min_threshold,
        )
    
    num_tokens = len(token2id)
    print("Number of tokens after filtering: ", num_tokens)
    

    print("-- Loading development set")
    dev_labels, dev_explanations, dev_premises, dev_hypotheses, dev_img_names, dev_original_explanations, _, _, dev_max_length, _ = \
        load_e_vsnli_dataset(
            FLAGS.dev_filename,
            token2id,
            label2id,
            buffer_size=FLAGS.buffer_size,
            padding_length=train_max_length,
        )

    if FLAGS.imbalance == True:
        dev_num_examples = dev_labels.shape[0]
        class_freqs = np.bincount(dev_labels) / dev_num_examples
        class_weights = 1 / (class_freqs * num_labels)
        print("Class frequencies: ", class_freqs)
        print("Weights: ", class_weights)
        np.save(FLAGS.model_save_filename + '_class_freqs.npy', class_freqs)
    print("-- Loading images")
    image_reader = ImageReader(FLAGS.img_names_filename, FLAGS.img_features_filename)
    
    
    print("-- Saving parameters")
    with open(FLAGS.model_save_filename + ".params", mode="w") as out_file:
        json.dump(vars(FLAGS), out_file)
        print("Params saved to: {}".format(FLAGS.model_save_filename + ".params"))

        with open(FLAGS.model_save_filename + ".index", mode="wb") as out_file:
            pickle.dump(
                {
                    "token2id": token2id,
                    "id2token": id2token,
                    "label2id": label2id,
                    "id2label": id2label
                },
                out_file
            )
            print("Index saved to: {}".format(FLAGS.model_save_filename + ".index"))
            
    model_config.set_vocab_size(num_tokens) 
    print("Vocab size, set to %d" % num_tokens)
    model_config.set_alpha(FLAGS.alpha)
    print("alpha = %f, set!" % FLAGS.alpha)
         
    num_examples = train_labels.shape[0]
    num_batches = num_examples // FLAGS.batch_size

    dev_num_examples = dev_labels.shape[0]
    dev_batches_indexes = np.arange(dev_num_examples)
    num_batches_dev = dev_num_examples // FLAGS.dev_batch_size   
           
    tf.reset_default_graph()
    
    # Build the TensorFlow graph and train it
    g = tf.Graph()
    with g.as_default():       
          
        model = build_model(model_config, embeddings, mode=mode)

        # Set up the learning rate.
        learning_rate_decay_fn = None
        learning_rate = tf.constant(training_config.initial_learning_rate)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (num_examples / FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                  return tf.train.exponential_decay(
                      learning_rate,
                      global_step,
                      decay_steps=decay_steps,
                      decay_rate=training_config.learning_rate_decay_factor,
                      staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model['total_loss'],
            global_step=model['global_step'],
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)
        
    
        dev_best_ppl = 1e8
        stopping_step = 0
        best_epoch = None
        should_stop = False

        # initialize all variables 
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            #session.run(tf.initializers.tables_initializer(name='init_all_tables'))
            
            t = 0 # counting iterations
            
            time_now = datetime.now()

            for epoch in range(training_config.total_num_epochs):
                if should_stop:
                    break

                print("\n==> Online epoch # {0}".format(epoch + 1))
                progress = Progbar(num_batches)
                batches_indexes = np.arange(num_examples)
                np.random.shuffle(batches_indexes)

                np.random.shuffle(batches_indexes)
                batch_index = 1
                loss_history = []
                epoch_loss = 0

                for indexes in batch(batches_indexes, FLAGS.batch_size):
                    
                    t += 1
                    batch_hypotheses = train_hypotheses[indexes]
                    batch_labels = train_labels[indexes]

                    # explanations have been encoded / padded when loaded
                    batch_explanations = train_explanations[indexes]
                    batch_explanation_lengths = [len(expl) for expl in batch_explanations]

                    batch_img_names = [train_img_names[i] for i in indexes]
                    batch_img_features = image_reader.get_features(batch_img_names)

                    total_loss_value = _step(session, batch_hypotheses, batch_labels, batch_explanations, batch_img_features, train_op, model, model_config.lstm_dropout_keep_prob) # run each training step 

                    progress.update(batch_index, [("Loss", total_loss_value)])
                    loss_history.append(total_loss_value)
                    epoch_loss += total_loss_value
                    batch_index += 1
                    
                    if FLAGS.print_every > 0 and t % FLAGS.print_every == 0:
                        print('(Iteration %d) loss: %f, and time elapsed: %.2f minutes' % (
                            t + 1, float(loss_history[-1]), (datetime.now() - time_now).seconds/60.0))

                print("Current mean training loss: {}\n".format(epoch_loss / num_batches))

                print("-- Validating model")

                progress = Progbar(num_batches_dev)

                dev_num_correct = 0           
                dev_batch_index = 0
                cum_dev_n_words = 0
                cum_dev_ppl = 0
                
                for indexes in batch(dev_batches_indexes, FLAGS.dev_batch_size):

                    t += 1
                    
                    dev_batch_num_correct = 0
                    
                    dev_batch_index += 1
                    dev_batch_hypotheses = dev_hypotheses[indexes]
                    dev_batch_labels = dev_labels[indexes]

                    # explanations have been encoded / padded when loaded
                    dev_batch_explanations = dev_explanations[indexes]
                    dev_batch_explanation_lengths = [len(expl) for expl in dev_batch_explanations]
                
                    dev_batch_img_names = [dev_img_names[i] for i in indexes]
                    dev_batch_img_features = image_reader.get_features(dev_batch_img_names)

                    explanation_loss_value = _run_validation(session, dev_batch_hypotheses, dev_batch_labels, dev_batch_explanations, dev_batch_img_features, len(indexes), ilabel2itoken, model, 1.0)

                    cum_dev_ppl += explanation_loss_value[0]
                    cum_dev_n_words += np.sum(dev_batch_explanation_lengths)

                    progress.update(dev_batch_index, [("Perplexity", np.exp(cum_dev_ppl/cum_dev_n_words))])
                    
            
                dev_ppl = np.exp(cum_dev_ppl/cum_dev_n_words)
                print("Current mean validation perplexity: {}".format(dev_ppl))
                
                
                #if True:
                if dev_ppl < dev_best_ppl:
                    stopping_step = 0
                    best_epoch = epoch + 1
                    dev_best_ppl = dev_ppl
                    model['saver'].save(session, FLAGS.model_save_filename + ".ckpt")
                    print("Best mean validation perplexity: {} (reached at epoch {})".format(dev_best_ppl, best_epoch))
                    print("Best model saved to: {}".format(FLAGS.model_save_filename))
                else:
                    stopping_step += 1
                    print("Current stopping step: {}".format(stopping_step))
                if stopping_step >= FLAGS.patience:
                    print("Early stopping at epoch {}!".format(epoch + 1))
                    print("Best mean validation perplexity: {} (reached at epoch {})".format(dev_best_ppl, best_epoch))
                    should_stop = True
                if epoch + 1 >= training_config.total_num_epochs:
                    print("Stopping at epoch {}!".format(epoch + 1))
                    print("Best mean validation perplexity: {} (reached at epoch {})".format(dev_best_ppl, best_epoch))

                

if __name__ == '__main__':
   
    parser = build_parser()
    
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

