import Utils
import pickle
import numpy as np
import soundfile as sf
import GenderDetectorTester
import GenderDetectorTrainer
from pathlib import Path


def split_wav_and_get_genders(wav, labelling, processed_wavs_directory):
    """This function will split the wav file with the labelling to obtain a wav for
    each of the distinct voices. With these wav files, it will also differentiate its gender,
    write it on the file's name, and return a dictionary that clarifies the gender of each
    speaker"""
    speech_dif_audios = {}
    for label in labelling:
        new_sub_wav = wav[int(label[1] * Utils.SAMPLING_RATE):int(label[2] * Utils.SAMPLING_RATE)]
        old_sub_wav = speech_dif_audios.get(label[0])
        if old_sub_wav is None:
            speech_dif_audios[label[0]] = new_sub_wav
        else:
            speech_dif_audios[label[0]] = np.concatenate((old_sub_wav, new_sub_wav))

    if not Path("pre-trained-gender-detectors/").is_dir():
        print("Training the gender detectors, it will take some minutes...")
        GenderDetectorTrainer.train_detectors()

    speaker_dictionary = {}
    male_voices = 0
    female_voices = 0
    for speaker_id, speaker_wav in speech_dif_audios.items():
        gender = GenderDetectorTester.test_wav_file(speaker_wav)
        if gender == Utils.MAN:
            new_speaker_id = gender + "_" + str(male_voices)
            male_voices += 1
        else:
            new_speaker_id = gender + "_" + str(female_voices)
            female_voices += 1
        sf.write(processed_wavs_directory + '/' + new_speaker_id + '.wav', speaker_wav,
                 Utils.SAMPLING_RATE, 'PCM_16')
        speaker_dictionary[speaker_id] = new_speaker_id
    sf.write(processed_wavs_directory + '/full_trimmed.wav', wav, Utils.SAMPLING_RATE, 'PCM_16')
    return speaker_dictionary


def split_wav_and_get_genders_from_raw(wav_file_path, labelling_file_path, processed_wavs_directory):
    """We use this function if we want to split the wav, but we have the raw objects,
    instead of reloading the creator"""
    with open(wav_file_path, 'rb') as wav_file:
        wav = pickle.load(wav_file)
    with open(labelling_file_path, 'rb') as labelling_file:
        labelling = pickle.load(labelling_file)
    split_wav_and_get_genders(wav, labelling, processed_wavs_directory)