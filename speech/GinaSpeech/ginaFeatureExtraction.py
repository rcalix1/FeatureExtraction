## feature extraction
## Gina - Speech Intelligent Agent
## www.rcalix.com
## Ricardo A. Calix, Ph.D.

#################################################################

import numpy as np
import pandas as pd

######################################################################

from GinaSpeech import GinaSpeech

######################################################################

gina = GinaSpeech()

######################################################################

gina.load_dictionary_gina() ## read from file

#gina.convert_wav_chunks_to_feature_vectors()

wav = 'audio/chunks/audio_chunk_yhid_ng5.wav'


gina.play_wav_from_file(wav)

gina.view_spectrogram(wav)
gina.view_stats_wave_file(wav)
gina.view_plotwave(wav)
gina.view_fftplot(wav)

text = "down"
gina.play_wav_chunk_from_text(text)

print( gina.dictionary_gina.items()  )

gina.write_dictionary_gina() # write to file

######################################################################

print("<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")
