import math

from resemblyzer import VoiceEncoder
from homemade_preprocesser import preprocess_wav
from pathlib import Path
import numpy as np
import pickle
from spectralcluster import SpectralClusterer

SAMPLING_RATE = 16000
encoder = VoiceEncoder()
# IMPORTANT, update the address with yours (I should do it in a relative way or with parameters, but I
# still haven't changed it
GLOBAL_FILE_ADDRESS = "/home/icel/Desktop/ENPC/Hackathon/Bechdel-test/resources/voxconverse/wav/hqyok.wav"


def consecutive_split(data: np.ndarray):
    splits = []
    count = 0
    current_frame = data[0]
    for frame in data:
        if frame != current_frame:
            splits.append((current_frame, count/SAMPLING_RATE))
            current_frame = frame
            #count = 0
        else:
            count += 1
    return splits


dif_labels = np.array([])
def get_ordered_label(label):
    global dif_labels
    if label not in dif_labels:
        dif_labels = np.append(dif_labels, label)
    number = dif_labels.tolist().index(label)
    return "%02d" % (number,)


def create_labelling(labels, wav_splits, audio_mask):
    times = [((s.start + s.stop) / 2) / SAMPLING_RATE for s in wav_splits]
    labelling = []
    #audio_mask = consecutive_split(np.array(audio_mask))
    speakers_time_list = []
    sound_counter = 0
    for i, sound in enumerate(audio_mask):
        if not sound:
            speakers_time_list.append("None")
        if sound:
            sound_counter += 1
            if math.floor(sound_counter/960) < len(labels):
                label = get_ordered_label(labels[math.floor(sound_counter/960)])
                speakers_time_list.append(label)

    print("Speakers list", consecutive_split(speakers_time_list))
    start_time = 0
    for i, time in enumerate(times):
        if i > 0 and labels[i] != labels[i - 1]:
            temp = [str(labels[i - 1]), start_time, time]
            labelling.append(tuple(temp))
            start_time = time
        if i == len(times) - 1:
            temp = [str(labels[i]), start_time, time]
            labelling.append(tuple(temp))
    return labelling


def main():

    #  The first time we run it, we should uncomment this part, because it will create the preprocesssed wav
    # and the labelling to work with. Then if we want to work with these
    #  First we take the sound file
    fpath = Path(GLOBAL_FILE_ADDRESS)

    #  We then preprocess it to get the processed wav file
    wav, audio_mask = preprocess_wav(fpath)

    # This cuts the audio in mel slices
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

    # This does the cluster part of distinguishing the participants
    clusterer = SpectralClusterer(
        min_clusters=1,
        max_clusters=100,
        #p_percentile=0.90, I commented these two lines because it gives an error but they are in the paper
        #gaussian_blur_sigma=1
        )

    labels = clusterer.predict(cont_embeds)
    labelling = create_labelling(labels, wav_splits, audio_mask)
    print(labelling)
    #  We save the raw objects (this is done to improve the speed of the debugging process, because all
    # these steps take a long time
    with open('rawObjects/labelling', 'wb') as labelling_file:
        pickle.dump(labelling, labelling_file)
    with open('rawObjects/wav', 'wb') as wav_file:
        pickle.dump(wav, wav_file)


if __name__ == "__main__":
    main()
