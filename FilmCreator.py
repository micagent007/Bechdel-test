from moviepy.editor import *
import gizeh as gz
from pydub import AudioSegment

import Utils

WHITE = (255, 255, 255)
VIDEO_SIZE = (640, 480)
BLUE = (59/255, 89/255, 152/255)
RED = (152/255, 20/255, 15/255)


def render_speaker(speaker):
    surface = gz.Surface(640, 60, bg_color=(1, 1, 1))
    color = BLUE
    if Utils.WOMAN in speaker:
        color = RED
    speaker_name = gz.text(speaker, fontfamily="Charter",
                           fontsize=30, fontweight='bold', fill=color, xy=(320, 40))
    speaker_name.draw(surface)
    return surface.get_npimage()


def create_film(processed_wavs_directory, labelling, speakers_dictionary):
    scenes = []
    for label in labelling:
        speaker = speakers_dictionary[label[0]]
        color = 'blue'
        if Utils.WOMAN in speaker:
            color = 'red'
        scene = TextClip(speaker, color=color, fontsize=64).set_start(label[1]).set_end(label[2]).set_pos('center')
        scenes.append(scene)

    video = CompositeVideoClip(scenes,
        size=VIDEO_SIZE). \
        on_color(
        color=WHITE,
        col_opacity=1).set_duration(labelling[len(labelling)-1][2])
    audio_clip = AudioFileClip(processed_wavs_directory + '/full_trimmed.wav')
    video = video.set_audio(audio_clip)
    video.write_videofile('processed_film.mp4', fps=10)
    return
