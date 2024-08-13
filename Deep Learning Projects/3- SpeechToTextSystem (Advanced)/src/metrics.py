import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint 
from jiwer import wer
import numpy as np
from keras.utils import register_keras_serializable


# from utils import load_data
# from model import model
from features_extraction import num_to_char

@register_keras_serializable()
def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def levenshtein_distance(y_true, y_pred):
    """
        Levenshtein distance is a string metric for measuring the difference
	    between two sequences. Informally, the levenshtein disctance is defined as
	    the minimum number of single-character edits (substitutions, insertions or
	    deletions) required to change one word into the other. We can naturally
	    extend the edits to word level when calculate levenshtein disctance for
	    two sentences.
    """
    m = len(y_true)
    n = len(y_pred)

    # special case
    if y_true == y_pred:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        y_true, y_pred = y_pred, y_true
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if y_true[i - 1] == y_pred[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


# # Define the checkpoint callback
# checkpoint_callback = ModelCheckpoint(
#     filepath='./Models/checkpoint_epoch_{epoch:02d}.keras',
#     save_weights_only=True,
#     save_freq='epoch',
#     verbose=1
# )

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


#Callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
            
        # Additional: Calculate Levenshtein distance for a sample
        sample_idx = np.random.randint(0, len(predictions))
        levenshtein_dist = levenshtein_distance(targets[sample_idx], predictions[sample_idx])
        print("-" * 100)
        print(f"Sample Levenshtein Distance: {levenshtein_dist}")
        print("-" * 100)

        # Print example targets and predictions
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)