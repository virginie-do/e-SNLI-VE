
"""Image-to-text model and training configurations."""

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""

        # Number of unique words in the vocab (plus 4, for <NULL>, <UNK>, <START>, <END>)
        # This one depends on your chosen vocab size in the preprocessing steps. Normally 
        # 5,000 might be a good choice since top 5,000 have covered most of the common words
        # appear in the data set. The rest not included in the vocab will be used as <UNK>
        #self.vocab_size = 5004 # actually the users sets this with the function _set_vocab_size()

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.image_feature_size = 2048  # equal to output layer size from inception v3
        self.num_lstm_units = 512
        #self.embedding_size = 512
        self.embedding_size = 300

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7

        # length of each explanation after padding 
        self.padded_length = 50

        # special wording
        self._null = 0 
        self._start = 2 # actually not using those attributes
        self._end = 3 #

        # config for VQA model
        self.num_img_features = 36 
        self.img_features_size = 2048

        self.train_embeddings = True
        self.classification_hidden_size = 512
        self.multimodal_fusion_hidden_size = 512

        self.num_labels = 3
        #self.alpha = 0.6
    
    def set_vocab_size(self, num_tokens):
        self.vocab_size = num_tokens
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    

    

class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Optimizer for training the model.
        self.optimizer = "Adam" # or "SGD"

        # Learning rate for the initial phase of training.
        #self.initial_learning_rate = 2.0
        self.initial_learning_rate = 0.001
        self.learning_rate_decay_factor = 0.5 
        self.num_epochs_per_decay = 8.0 

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        #self.total_num_epochs = 5
        self.total_num_epochs = 100
