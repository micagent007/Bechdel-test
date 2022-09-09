import numpy as np
import warnings
import os
import Utils
from resemblyzer import preprocess_wav
from sklearn.mixture import GaussianMixture as GMM
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

#  Here we have to put the training data (relative to the file)
# This has to be a folder with .wav files of the male/female voices
# respectively. It can be downloaded from the drive of readme
male_training_source = "resources/gender_detector_dataset/male_clips"
female_training_source = "resources/gender_detector_dataset/female_clips"


def get_features(source):
    # Split files
    files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.wav')]
    len_train = int(len(files) * 0.8)
    train_files = files[:len_train]
    test_files = files[len_train:]

    # Train features
    features_train = []
    for f in train_files:
        wav = preprocess_wav(f)
        vector = Utils.ENCODER.embed_utterance(wav, return_partials=False, rate=16)
        if len(features_train) == 0:
            features_train = vector
        else:
            features_train = np.vstack((features_train, vector))

    # Test features
    features_test = []
    for f in test_files:
        wav = preprocess_wav(f)
        vector = Utils.ENCODER.embed_utterance(wav, return_partials=False, rate=16)
        if len(features_test) == 0:
            features_test = vector
        else:
            features_test = np.vstack((features_test, vector))

    return features_train, features_test


def train_detectors():
    """This method will create the gender detectors and place them as raw objects in a folder.
    For it to work, there has to be the resources/gender_detector_dataset folder in the same folder of the
    python code. The AudioSet can be downloaded from the readme drive URL"""
    print("Training male detector...")
    features_train_male, features_test_male = get_features(male_training_source)
    gmm_male = GMM(n_components=8, max_iter=200, covariance_type='diag', n_init=3)
    gmm_male.fit(features_train_male)

    print("Training female detector...")
    features_train_female, features_test_female = get_features(female_training_source)
    gmm_female = GMM(n_components=8, max_iter=200, covariance_type='diag', n_init=3)
    gmm_female.fit(features_train_female)

    # Here we test the effectivity of the gmm trained
    output = []
    for f in features_test_male:
        log_likelihood_male = np.array(gmm_male.score([f])).sum()
        log_likelihood_female = np.array(gmm_female.score([f])).sum()
        if log_likelihood_male > log_likelihood_female:
            output.append(0)
        else:
            output.append(1)
    accuracy_male = (1 - sum(output) / len(output))
    print("The estimated accuracy is:", accuracy_male)

    # We finish by saving the models with pickle to reuse
    pickle.dump(gmm_male, open("pre-trained-gender-detectors/male.gmm", "wb"))
    pickle.dump(gmm_female, open("pre-trained-gender-detectors/female.gmm", "wb"))
