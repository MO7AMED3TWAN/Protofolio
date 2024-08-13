import pandas as pd
import os
import tensorflow as tf
from features_extraction import *

def load_data(train_csv, train_audio_dir, adapt_csv, adapt_audio_dir):
    # Read CSV files into DataFrames
    train_df = pd.read_csv(train_csv)
    adapt_df = pd.read_csv(adapt_csv)

    # Add audio_path column to DataFrames
    train_df['audio_path'] = train_df['audio'].apply(lambda x: os.path.join(train_audio_dir, x + '.wav'))
    adapt_df['audio_path'] = adapt_df['audio'].apply(lambda x: os.path.join(adapt_audio_dir, x + '.wav'))

    split = int(len(train_df) * 0.80)
    df_train = train_df[:split]
    df_val   = train_df[split:]
    # Here we add adapt data to the split sample that
    # we get it from train File

    # Define the batch size.
    batch_size = 32


    # Assume df_train is a DataFrame containing the audio paths and transcripts.
    # Example: df_train = pd.DataFrame({"audio_path": ["path1.wav", "path2.wav"], "transcript": ["text1", "text2"]})
    # Note: Ensure df_train is defined and contains the columns "audio_path" and "transcript".
    
    # Define the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["audio_path"]), list(df_train["transcript"]))
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["audio_path"]), list(df_val["transcript"]))
    )
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset, df_train, df_val