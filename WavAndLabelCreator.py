import pandas as pd
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.cluster import KMeans
from pathlib import Path
from OptimalCluster.opticlust import Optimal

import pickle

SAMPLING_RATE = 16000
encoder = VoiceEncoder()
# IMPORTANT, update the address with yours (I should do it in a relative way or with parameters, but I
# still haven't changed it
GLOBAL_FILE_ADDRESS = "C:\\Users\\micah\\Desktop\\Hackaton 2022\\Bechdel-test\\audio_res\\il-sappelle-just-leblanc.wav"


def create_labelling(labels, wav_splits):
    times = [((s.start + s.stop) / 2) / SAMPLING_RATE for s in wav_splits]
    labelling = []
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
    wav = preprocess_wav(fpath)

    # This cuts the audio in mel slices
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

    # This does the cluster part of distinguishing the participants

    #Here we're looking for the number of clusters
    opt = Optimal()
    df=pd.DataFrame(cont_embeds)
    opt_value=opt.silhouette(df)

    #Use of Kmeans with the right amount of clusters
    clusterer=KMeans(n_clusters=opt_value)
    labels=clusterer.fit_predict(cont_embeds)
    labelling = create_labelling(labels, wav_splits)


    #  We save the raw objects (this is done to improve the speed of the debugging process, because all
    # these steps take a long time
    with open('rawObjects/labelling', 'wb') as labelling_file:
        pickle.dump(labelling, labelling_file)
    with open('rawObjects/wav', 'wb') as wav_file:
        pickle.dump(wav, wav_file)


if __name__ == "__main__":
    main()
