import os
import tensorflow as tf 
from model import model
from utils import load_data
from metrics import  CallbackEval # checkpoint_callback not added


# Now, calling the load_data function
train_csv = "./Data/train.csv"  # Replace with the actual path to your CSV file
train_audio_dir = "./Data/Clean_Data/"  # Replace with the directory containing your audio files

adapt_csv = "./Data/adapt.csv"  # Replace with the actual path to your CSV file
adapt_audio_dir = "./Data/Clean_Data/"  # Replace with the directory containing your audio files

train_dataset, validation_dataset, df_train, df_val = load_data(train_csv, train_audio_dir, adapt_csv, adapt_audio_dir)


# Create an instance of the CallbackEval
callback_eval = CallbackEval(dataset=validation_dataset)  # Use your validation dataset here


# checkpoint_dir = './Models'
# checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_??.keras') ## Put Number Of Your Stop Epoch Here ??

# if os.path.exists(checkpoint_path):
#     print(f'Loading weights from {checkpoint_path}')
#     model.load_weights(checkpoint_path)
#     initial_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
# else:
#     print(f'Checkpoint file {checkpoint_path} not found. Starting training from scratch.')
#     initial_epoch = 0
    
# Define the number of epochs.
num_epochs = 50

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)

# Train the model, resuming from the last saved epoch if applicable
model.fit(
    train_dataset,
    epochs=num_epochs,
    initial_epoch=initial_epoch,
    callbacks=[callback_eval] # checkpoint_callback not added 
    )