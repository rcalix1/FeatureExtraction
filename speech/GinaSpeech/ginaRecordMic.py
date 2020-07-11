## Gina - Speech Intelligent Agent
## www.rcalix.com
## Ricardo A. Calix, Ph.D.

#################################################################

import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join

######################################################################

from GinaSpeech import GinaSpeech

######################################################################

gina = GinaSpeech()

######################################################################

gina.load_dictionary_gina()

temp_fname = gina.record_wav_utterance()
print("name is ", temp_fname)
gina.play_wav_from_file(  join(gina.path_utterances,temp_fname)  )

output_wav_chunks = gina.split_wav_to_chunks(join(gina.path_utterances, temp_fname))
print(output_wav_chunks)

for chunk in output_wav_chunks:
    gina.play_wav_from_file(   join(gina.path_chunks,chunk)  )
    gina.add_wav_chunk_to_gina_dictionary(chunk)
    print(chunk)
    print("************************************************")

gina.annotate_speech_chunk_files()

# run when you have new wav chunks that
# are currently not in the dictionary
gina.add_new_wav_chunks_to_dictionary()

print( gina.dictionary_gina.items()  )

text = "elephant"
gina.play_wav_chunk_from_text(text)

gina.write_dictionary_gina()

######################################################################

print("<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")
