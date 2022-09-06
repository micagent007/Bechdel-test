from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from spectralcluster import SpectralClusterer

sampling_rate = 16000


def create_labelling(labels, wav_splits):
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
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
    print("Starts process")
    fpath = Path("/home/icel/Desktop/ENPC/Hackathon Bechdel/audio_res/male.wav")
    wav = preprocess_wav(fpath)
    encoder = VoiceEncoder()
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

    print("Preprocesses")
    print(cont_embeds.shape)

    clusterer = SpectralClusterer(
        min_clusters=2,
        max_clusters=100,
        #p_percentile=0.90,
        #gaussian_blur_sigma=1
        )

    labels = clusterer.predict(cont_embeds)
    print("Created labels")

    labelling = create_labelling(labels, wav_splits)

    print("Labeled")
    print(labelling)

    print("Program finished")


if __name__ == "__main__":
    main()
