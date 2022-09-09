import WavAndLabellingCreator
import WavAndLabellingProcessor
import FilmCreator
import os


# Good example: resources/voxconverse/wav/ccokr.wav
# Here we give the address of the wav file to process
wav_file_address = "/home/icel/Desktop/ENPC/Hackathon/Bechdel-test/resources/random_audios_for_testing/many_voices_short.wav"

# Here we give the address where we want the different voices wav files
processed_wavs_directory = os.path.abspath("resources/processed_audio_res")


def main():
    wav, labelling = WavAndLabellingCreator.create_wav_and_label(wav_file_address)
    speakers_dictionary = WavAndLabellingProcessor.split_wav_and_get_genders(wav, labelling, processed_wavs_directory)
    FilmCreator.create_film(processed_wavs_directory, labelling, speakers_dictionary)
    return


if __name__ == "__main__":
    main()
