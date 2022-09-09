import numpy as np
import warnings
import pickle

import Utils

warnings.simplefilter(action='ignore', category=FutureWarning)

# Here we write the relative path to the test wav file
wav_source = "resources/moviesoundclips.net/moviesoundclips.net/2.wav"

#   Here we write the relative path to the gmm sources
# (the ones we got with GenderDetectorTrainer)
gmm_male_source = 'pre-trained-gender-detectors/male.gmm'
gmm_female_source = 'pre-trained-gender-detectors/female.gmm'


def get_gmm(gmm_source):
    with open(gmm_source, 'rb') as gmm_file:
        return pickle.load(gmm_file)


def process_file(wav):
    vector = Utils.ENCODER.embed_utterance(wav, return_partials=False, rate=16)
    return vector


def test_wav_file(wav):
    embeded = process_file(wav)
    gmm_male = get_gmm(gmm_male_source)
    gmm_female = get_gmm(gmm_female_source)
    log_likelihood_male = np.array(gmm_male.score([embeded])).sum()
    log_likelihood_female = np.array(gmm_female.score([embeded])).sum()
    if log_likelihood_male > log_likelihood_female:
        return Utils.MAN
    else:
        return Utils.WOMAN
