## speech to text
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
gina.load_dictionary_phonemes()

gina.build_vector_space_phonemes()


print("start speaking your utterance  ")
temp_fname = gina.record_wav_utterance()
print("name is ", temp_fname)

gina.play_wav_from_file(  join(gina.path_utterances,temp_fname)  )

output_wav_chunks = gina.split_wav_to_chunks(join(gina.path_utterances, temp_fname))
print(output_wav_chunks)


for chunk in output_wav_chunks:
    gina.play_wav_from_file(   join(gina.path_chunks,chunk)  )
    gina.add_wav_chunk_to_gina_dictionary(chunk)
    print(chunk)
    predicted_text = gina.predict_text_from_chunk(join(gina.path_chunks,chunk))
    print("predicted text from chunk: ", predicted_text)
    print("************************************************")


gina.write_dictionary_phonemes()
gina.write_dictionary_gina()

######################################################################

print("<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")

