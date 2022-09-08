import pandas as pd
from resemblyzer import preprocess_wav, VoiceEncoder
from sklearn.cluster import KMeans
from pathlib import Path
from OptimalCluster.opticlust import Optimal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import pickle

SAMPLING_RATE = 16000
encoder = VoiceEncoder()
# IMPORTANT, update the address with yours (I should do it in a relative way or with parameters, but I
# still haven't changed it
GLOBAL_FILE_ADDRESS = "C:\\Users\\micah\\Desktop\\Hackaton 2022\\Bechdel-test\\audio_res\\un-gars-une-fille-best-of-jalousie.wav"


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

    #Here we're looking for the optimal number of clusters i.e. the amount of people talking

    #we create the set of data from mel's spectrogram
    opt = Optimal({'max_iter':200})
    df=pd.DataFrame(cont_embeds)

    #we normalize this set of data in order to give an equal weight to each data
    scalar = StandardScaler()
    df_scaled = pd.DataFrame(scalar.fit_transform(df),columns=df.columns)

    #we project the set of data that are currently on a 256 dimensionnal space to a
    #2D plane in order to visualize the gathering of data
    pca = PCA(n_components=2)
    df_pca=pd.DataFrame(pca.fit_transform(df_scaled))
    plt.scatter(df_pca[0],df_pca[1],marker="+")
    plt.show()

    #using this new 2D set of data we're able to compute the optimal
    #amount of cluster i.e. the amount of characters in the discussion
    opt_value1=opt.elbow(df_pca)

    #Use of Kmeans with the right amount of clusters
    clusterer=KMeans(n_clusters=opt_value1)
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
