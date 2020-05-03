"""Class for generating captions from an image-to-text model.
   This is based on Google's https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/caption_generator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
import math

import tensorflow as tf

import numpy as np

### Credits ###
# https://github.com/aaxwaz/Image-Captioning-Model-in-TensorFlow

class Explanation(object):
  """Represents a complete or partial explanation."""

  def __init__(self, sentence, state, logprob, score, metadata=None):
    """Initializes the Caption.
    Args:
      sentence: List of word ids in the explanation.
      state: Model state after generating the previous word.
      logprob: Log-probability of the explanation.
      score: Score of the explanation.
      metadata: Optional metadata associated with the partial sentence. If not
        None, a list of strings with the same length as 'sentence'.
    """
    self.sentence = sentence
    self.state = state
    self.logprob = logprob
    self.score = score
    self.metadata = metadata

  def __cmp__(self, other):
    """Compares Explanation by score."""
    assert isinstance(other, Explanation)
    if self.score == other.score:
      return 0
    elif self.score < other.score:
      return -1
    else:
      return 1
  
  # For Python 3 compatibility (__cmp__ is deprecated).
  def __lt__(self, other):
    assert isinstance(other, Explanation)
    return self.score < other.score
  
  # Also for Python 3 compatibility.
  def __eq__(self, other):
    assert isinstance(other, Explanation)
    return self.score == other.score


class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n):
    self._n = n
    self._data = []

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False):
    """Extracts all elements from the TopN. This is a destructive operation.
    The only method that can be called immediately after extract() is reset().
    Args:
      sort: Whether to return the elements in descending sorted order.
    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    self._data = None
    if sort:
      data.sort(reverse=True)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []


class LabelExplanationGenerator(object):
  """Class to generate explanations from an image-and-sentence-to-text model."""

  def __init__(self,
               model,
               vocab,
               ilabel2itoken,
               beam_size=3, 
               max_explanation_length=24,
               length_normalization_factor=0.0):
    """Initializes the generator.
    Args:
      model: Object encapsulating a trained image-and-sentence-to-text model. Must have
        methods feed_image() and inference_step(). For example, an instance of
        InferenceWrapperBase.
      vocab: A Vocabulary object.
      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.
    """
    self.vocab = vocab
    self.ilabel2itoken = ilabel2itoken
    self.model = model
    
    self.beam_size = beam_size
    self.max_explanation_length = max_explanation_length
    self.length_normalization_factor = length_normalization_factor
  
    
  def _feed_image(self, sess, hypothesis, img_feature):
    # get initial state using image feature 
    feed_dict = {self.model['hypothesis_input']: hypothesis, 
                 self.model['img_features_input']: img_feature, 
                 self.model['dropout_input']: 1.0,
                 self.model['keep_prob']: 1.0}
    label, state = sess.run([tf.argmax(self.model['label_logits'], axis=1), self.model['initial_state']], feed_dict=feed_dict)
    return label, state
    
  def _inference_step(self, sess, input_feed_list, state_feed_list, max_explanation_length):
  
    mask = np.zeros((1, max_explanation_length))
    mask[:, 0] = 1
    softmax_outputs = []
    new_state_outputs = []
    
    for input, state in zip(input_feed_list, state_feed_list):
        feed_dict={self.model['input_seqs']: input, 
                   self.model['initial_state']: state, 
                   self.model['input_mask']: mask, 
                   self.model['dropout_input']: 1.0,
                   self.model['keep_prob']: 1.0}
        softmax, new_state = sess.run([self.model['softmax'], self.model['final_state']], feed_dict=feed_dict)
        softmax_outputs.append(softmax)
        new_state_outputs.append(new_state)
        
    return softmax_outputs, new_state_outputs, None

  def beam_search(self, sess, hypothesis, img_feature):
    """Runs beam search explanation generation on a single image-sentence pair.
    Args:
      sess: TensorFlow Session object.
      hypothesis: encoded hypothesis sentence
      feature: extracted bottom-up feature of one image.
    Returns:
      A list of Caption sorted by descending score.
    """
    # Feed in the image to get the initial state.
    label, initial_state = self._feed_image(sess, hypothesis, img_feature)
    # TODO: prepend label

    initial_beam = Explanation(
        sentence=[self.ilabel2itoken[label[0]]], #TODO: prepend label instead of using <start>
        state=initial_state,
        logprob=0.0,
        score=0.0,
        metadata=[""])
    partial_explanations = TopN(self.beam_size)
    partial_explanations.push(initial_beam)
    complete_explanations = TopN(self.beam_size)

    # Run beam search.
    for t in range(self.max_explanation_length - 1):
      if t == 0:      
        initial_beam = Explanation(
            sentence=[self.vocab['<start>']], 
            state=initial_state,
            logprob=0.0,
            score=0.0,
            metadata=[""])
        partial_explanations = TopN(self.beam_size)
        partial_explanations.push(initial_beam)
        complete_explanations = TopN(self.beam_size)
      if t == 1:      
        initial_beam = Explanation(
            sentence=[self.ilabel2itoken[label[0]]], #TODO: prepend label instead of using <start>
            state=initial_state,
            logprob=0.0,
            score=0.0,
            metadata=[""])
        partial_explanations = TopN(self.beam_size)
        partial_explanations.push(initial_beam)
        complete_explanations = TopN(self.beam_size)
      partial_explanations_list = partial_explanations.extract()
      partial_explanations.reset()
      input_feed = [np.array([c.sentence[-1]]).reshape(1, 1) for c in partial_explanations_list]
      state_feed = [c.state for c in partial_explanations_list]

      softmax, new_states, metadata = self._inference_step(sess,
                                                           input_feed,
                                                           state_feed, 
                                                           self.max_explanation_length)

      for i, partial_explanation in enumerate(partial_explanations_list):
        word_probabilities = softmax[i][0]
        state = new_states[i]
        # For this partial explanation, get the beam_size most probable next words.
        words_and_probs = list(enumerate(word_probabilities))
        words_and_probs.sort(key=lambda x: -x[1])
        words_and_probs = words_and_probs[0:self.beam_size]
        # Each next word gives a new partial explanation.
        for w, p in words_and_probs:
          if p < 1e-12:
            continue  # Avoid log(0).
          sentence = partial_explanation.sentence + [w]
          logprob = partial_explanation.logprob + math.log(p)
          score = logprob
          if metadata:
            metadata_list = partial_explanation.metadata + [metadata[i]]
          else:
            metadata_list = None
          if w == self.vocab['<end>']: 
            if self.length_normalization_factor > 0:
              score /= len(sentence)**self.length_normalization_factor
            beam = Explanation(sentence, state, logprob, score, metadata_list)
            complete_explanations.push(beam)
          else:
            beam = Explanation(sentence, state, logprob, score, metadata_list)
            partial_explanations.push(beam)
      if partial_explanations.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_explanations.size():
      complete_explanations = partial_explanations

    return label, complete_explanations.extract(sort=True)




class ExplanationGenerator(object):
  """Class to generate explanations from an image-and-sentence-to-text model."""

  def __init__(self,
               model,
               vocab,
               beam_size=3,
               max_explanation_length=24,
               length_normalization_factor=0.0):
    """Initializes the generator.
    Args:
      model: Object encapsulating a trained image-and-sentence-to-text model. Must have
        methods feed_image() and inference_step(). For example, an instance of
        InferenceWrapperBase.
      vocab: A Vocabulary object.
      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.
    """
    self.vocab = vocab
    self.model = model
    
    self.beam_size = beam_size
    self.max_explanation_length = max_explanation_length
    self.length_normalization_factor = length_normalization_factor
  
    
  def _feed_image(self, sess, hypothesis, img_feature):
    # get initial state using image feature 
    feed_dict = {self.model['hypothesis_input']: hypothesis, 
                 self.model['img_features_input']: img_feature, 
                 self.model['dropout_input']: 1.0,
                 self.model['keep_prob']: 1.0}
    state = sess.run(self.model['initial_state'], feed_dict=feed_dict)
    return state
    
  def _inference_step(self, sess, input_feed_list, state_feed_list, max_explanation_length):
  
    mask = np.zeros((1, max_explanation_length))
    mask[:, 0] = 1
    softmax_outputs = []
    new_state_outputs = []
    
    for input, state in zip(input_feed_list, state_feed_list):
        feed_dict={self.model['input_seqs']: input, 
                   self.model['initial_state']: state, 
                   self.model['input_mask']: mask, 
                   self.model['dropout_input']: 1.0,
                   self.model['keep_prob']: 1.0}
        softmax, new_state = sess.run([self.model['softmax'], self.model['final_state']], feed_dict=feed_dict)
        softmax_outputs.append(softmax)
        new_state_outputs.append(new_state)
        
    return softmax_outputs, new_state_outputs, None

  def beam_search(self, sess, hypothesis, img_feature):
    """Runs beam search explanation generation on a single image-sentence pair.
    Args:
      sess: TensorFlow Session object.
      hypothesis: encoded hypothesis sentence
      feature: extracted bottom-up feature of one image.
    Returns:
      A list of Caption sorted by descending score.
    """
    # Feed in the image to get the initial state.
    initial_state = self._feed_image(sess, hypothesis, img_feature)

    # Run beam search.
    for t in range(self.max_explanation_length - 1):
      if t == 0:      
        initial_beam = Explanation(
            sentence=[self.vocab['<start>']], 
            state=initial_state,
            logprob=0.0,
            score=0.0,
            metadata=[""])
        partial_explanations = TopN(self.beam_size)
        partial_explanations.push(initial_beam)
        complete_explanations = TopN(self.beam_size)

      partial_explanations_list = partial_explanations.extract()
      partial_explanations.reset()
      input_feed = [np.array([c.sentence[-1]]).reshape(1, 1) for c in partial_explanations_list]
      state_feed = [c.state for c in partial_explanations_list]

      softmax, new_states, metadata = self._inference_step(sess,
                                                           input_feed,
                                                           state_feed, 
                                                           self.max_explanation_length)

      for i, partial_explanation in enumerate(partial_explanations_list):
        word_probabilities = softmax[i][0]
        state = new_states[i]
        # For this partial explanation, get the beam_size most probable next words.
        words_and_probs = list(enumerate(word_probabilities))
        words_and_probs.sort(key=lambda x: -x[1])
        words_and_probs = words_and_probs[0:self.beam_size]
        # Each next word gives a new partial explanation.
        for w, p in words_and_probs:
          if p < 1e-12:
            continue  # Avoid log(0).
          sentence = partial_explanation.sentence + [w]
          logprob = partial_explanation.logprob + math.log(p)
          score = logprob
          if metadata:
            metadata_list = partial_explanation.metadata + [metadata[i]]
          else:
            metadata_list = None
          if w == self.vocab['<end>']: 
            if self.length_normalization_factor > 0:
              score /= len(sentence)**self.length_normalization_factor
            beam = Explanation(sentence, state, logprob, score, metadata_list)
            complete_explanations.push(beam)
          else:
            beam = Explanation(sentence, state, logprob, score, metadata_list)
            partial_explanations.push(beam)
      if partial_explanations.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_explanations.size():
      complete_explanations = partial_explanations

    return complete_explanations.extract(sort=True)




class AttentionExplanationGenerator(object):
  """Class to generate explanations from an image-and-sentence-to-text model."""

  def __init__(self,
               model,
               vocab,
               beam_size=3,
               max_explanation_length=24,
               length_normalization_factor=0.0):
    """Initializes the generator.
    Args:
      model: Object encapsulating a trained image-and-sentence-to-text model. Must have
        methods feed_image() and inference_step(). For example, an instance of
        InferenceWrapperBase.
      vocab: A Vocabulary object.
      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.
    """
    self.vocab = vocab
    self.model = model
    
    self.beam_size = beam_size
    self.max_explanation_length = max_explanation_length
    self.length_normalization_factor = length_normalization_factor
  
    
  def _feed_image(self, sess, hypothesis, img_feature):
    # get initial state using image feature 
    feed_dict = {self.model['hypothesis_input']: hypothesis, 
                 self.model['img_features_input']: img_feature, 
                 self.model['dropout_input']: 1.0,
                 self.model['keep_prob']: 1.0}
    #attn, state = sess.run([tf.nn.top_k(self.model['attention_output'],3).indices, self.model['initial_state']], feed_dict=feed_dict)
    attn, state = sess.run([self.model['attention_output'], self.model['initial_state']], feed_dict=feed_dict)
    return attn, state
    
  def _inference_step(self, sess, input_feed_list, state_feed_list, max_explanation_length):
  
    mask = np.zeros((1, max_explanation_length))
    mask[:, 0] = 1
    softmax_outputs = []
    new_state_outputs = []
    
    for input, state in zip(input_feed_list, state_feed_list):
        feed_dict={self.model['input_seqs']: input, 
                   self.model['initial_state']: state, 
                   self.model['input_mask']: mask, 
                   self.model['dropout_input']: 1.0,
                   self.model['keep_prob']: 1.0}
        softmax, new_state = sess.run([self.model['softmax'], self.model['final_state']], feed_dict=feed_dict)
        softmax_outputs.append(softmax)
        new_state_outputs.append(new_state)
        
    return softmax_outputs, new_state_outputs, None

  def beam_search(self, sess, hypothesis, img_feature):
    """Runs beam search explanation generation on a single image-sentence pair.
    Args:
      sess: TensorFlow Session object.
      hypothesis: encoded hypothesis sentence
      feature: extracted bottom-up feature of one image.
    Returns:
      A list of Caption sorted by descending score.
    """
    # Feed in the image to get the initial state.
    attn, initial_state = self._feed_image(sess, hypothesis, img_feature)

    # Run beam search.
    for t in range(self.max_explanation_length - 1):
      if t == 0:      
        initial_beam = Explanation(
            sentence=[self.vocab['<start>']], 
            state=initial_state,
            logprob=0.0,
            score=0.0,
            metadata=[""])
        partial_explanations = TopN(self.beam_size)
        partial_explanations.push(initial_beam)
        complete_explanations = TopN(self.beam_size)

      partial_explanations_list = partial_explanations.extract()
      partial_explanations.reset()
      input_feed = [np.array([c.sentence[-1]]).reshape(1, 1) for c in partial_explanations_list]
      state_feed = [c.state for c in partial_explanations_list]

      softmax, new_states, metadata = self._inference_step(sess,
                                                           input_feed,
                                                           state_feed, 
                                                           self.max_explanation_length)

      for i, partial_explanation in enumerate(partial_explanations_list):
        word_probabilities = softmax[i][0]
        state = new_states[i]
        # For this partial explanation, get the beam_size most probable next words.
        words_and_probs = list(enumerate(word_probabilities))
        words_and_probs.sort(key=lambda x: -x[1])
        words_and_probs = words_and_probs[0:self.beam_size]
        # Each next word gives a new partial explanation.
        for w, p in words_and_probs:
          if p < 1e-12:
            continue  # Avoid log(0).
          sentence = partial_explanation.sentence + [w]
          logprob = partial_explanation.logprob + math.log(p)
          score = logprob
          if metadata:
            metadata_list = partial_explanation.metadata + [metadata[i]]
          else:
            metadata_list = None
          if w == self.vocab['<end>']: 
            if self.length_normalization_factor > 0:
              score /= len(sentence)**self.length_normalization_factor
            beam = Explanation(sentence, state, logprob, score, metadata_list)
            complete_explanations.push(beam)
          else:
            beam = Explanation(sentence, state, logprob, score, metadata_list)
            partial_explanations.push(beam)
      if partial_explanations.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_explanations.size():
      complete_explanations = partial_explanations

    return attn, complete_explanations.extract(sort=True)
