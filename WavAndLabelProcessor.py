from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from spectralcluster import SpectralClusterer
import pickle
import numpy as np
import soundfile as sf

SAMPLING_RATE = 16000
# IMPORTANT, update the address with yours (I should do it in a relative way or with parameters, but I
# still haven't changed it
GLOBAL_FILE_ADDRESS = "C:\\Users\\micah\\Desktop\\Hackaton 2022\\Bechdel-test\\processed_audio_res\\"


def main():

    # We first work with the precreated files (with WavAndLabelCreator.py)
    with open('rawObjects/labelling', 'rb') as labelling_file:
        labelling = pickle.load(labelling_file)
    with open('rawObjects/wav', 'rb') as wav_file:
        wav = pickle.load(wav_file)

    speech_dif_audios = {}
    for label in labelling:
        new_sub_wav = wav[int(label[1] * SAMPLING_RATE):int(label[2] * SAMPLING_RATE)]
        old_sub_wav = speech_dif_audios.get(label[0])
        if old_sub_wav is None:
            speech_dif_audios[label[0]] = new_sub_wav
        else:
            speech_dif_audios[label[0]] = np.concatenate((old_sub_wav, new_sub_wav))

    c = 0
    for speech in speech_dif_audios.values():
        sf.write(GLOBAL_FILE_ADDRESS + 'many_voices_short_' + str(c) + '.wav', speech, 16000, 'PCM_16')
        c += 1


if __name__ == "__main__":
    main()
