import numpy as np
import warnings
import os
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.mixture import GaussianMixture as GMM
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
encoder = VoiceEncoder()

#  Here we have to put the training data (relative to the file)
# This has to be a folder with .wav files of the male/female voices
# respectively
male_training_source = "resources/AudioSet/male_clips"
female_training_source = "resources/AudioSet/female_clips"


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
        vector = encoder.embed_utterance(wav, return_partials=False, rate=16)
        if len(features_train) == 0:
            features_train = vector
        else:
            features_train = np.vstack((features_train, vector))

    # Test features
    features_test = []
    for f in test_files:
        wav = preprocess_wav(f)
        vector = encoder.embed_utterance(wav, return_partials=False, rate=16)
        if len(features_test) == 0:
            features_test = vector
        else:
            features_test = np.vstack((features_test, vector))

    return features_train, features_test


print("Training male detector")
features_train_male, features_test_male = get_features(male_training_source)
gmm_male = GMM(n_components=8, max_iter=200, covariance_type='diag', n_init=3)
gmm_male.fit(features_train_male)

print("Training female detector")
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
accuracy_male = (1 - sum(output)/len(output))
print("Accuracy:", accuracy_male)

# We finish by saving the models with pickle to reuse
pickle.dump(gmm_male, open("pre-trained-gender-detectors/male.gmm", "wb"))
pickle.dump(gmm_female, open("pre-trained-gender-detectors/female.gmm", "wb"))
