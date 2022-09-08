import numpy as np
import warnings
import os
from resemblyzer import preprocess_wav, VoiceEncoder
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
encoder = VoiceEncoder()

# Here we write the absolute path to the test wav file
wav_source = "/home/icel/Desktop/ENPC/Bachdel-test-data/moviesoundclips.net/moviesoundclips.net/626.wav"

#   Here we write the relative path to the gmm sources
# (the ones we got with GenderDetectorTrainer)
gmm_male_source = 'pre-trained-gender-detectors/male.gmm'
gmm_female_source = 'pre-trained-gender-detectors/female.gmm'


def get_gmm(gmm_source):
    with open(gmm_source, 'rb') as gmm_file:
        return pickle.load(gmm_file)


def process_file(source):
    wav = preprocess_wav(source)
    vector = encoder.embed_utterance(wav, return_partials=False, rate=16)
    return vector


embeded = process_file(wav_source)
gmm_male = get_gmm(gmm_male_source)
gmm_female = get_gmm(gmm_female_source)
log_likelihood_male = np.array(gmm_male.score([embeded])).sum()
log_likelihood_female = np.array(gmm_female.score([embeded])).sum()

# Here we see which one is more likely
if log_likelihood_male > log_likelihood_female:
    print("C'est un homme")
else:
    print("C'est une femme")

