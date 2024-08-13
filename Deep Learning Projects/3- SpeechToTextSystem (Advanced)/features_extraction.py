import tensorflow as tf
from tensorflow import keras

# Define the set of characters accepted in the transcription, including space.
characters = ' ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهةوىي'

# Create a StringLookup layer for character to number conversion.
char_to_num = keras.layers.StringLookup(vocabulary=list(characters), oov_token="")

# Create a StringLookup layer for number to character conversion (inverse lookup).
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Define constants for STFT.
frame_length = 256
frame_step = 160
fft_length = 384

def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    ###########################################
    ##  Process the label
    ###########################################
    
    # 7. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 8. Map the characters in label to numbers
    label = char_to_num(label)

    # 9. Return the spectrogram and label
    return spectrogram, label

# Function to preprocess a single audio file
def preprocess_audio(wav_file):
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    
    # Check if the audio length is sufficient
    if tf.shape(audio)[0] < fft_length:
        # Pad the audio signal to have the minimum length
        pad_amount = fft_length - tf.shape(audio)[0]
        audio = tf.pad(audio, paddings=[[0, pad_amount]])
    
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
   
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
   
    # 6. Normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
    return spectrogram