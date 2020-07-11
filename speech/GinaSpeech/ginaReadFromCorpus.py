## Gina Speech Intelligent Agent AI
## www.rcalix.com
## Ricardo A. Calix, Ph.D
## Read data from corpus

#################################################################

import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import string
import random
import shutil

######################################################################

from GinaSpeech import GinaSpeech

######################################################################

gina = GinaSpeech()

######################################################################

gina.load_dictionary_gina() ## read from file


#wav = 'audio/chunks/audio_chunk_yhid_ng5.wav'
#gina.play_wav_from_file(wav)

##########################################################
## the corpus

#path = "/home/rcalix1/Downloads/speech_commands/"

###########################################################

list_dirs = os.listdir(path)
print(list_dirs)

for dir in list_dirs:
    print(dir)
    new_path = join(path, dir)
    for f in os.listdir(new_path):
        yet_another_path = os.path.join(new_path, f)
        if os.path.isfile(  yet_another_path  ):
            print(yet_another_path)
            print(dir, " ", f)
            #gina.play_wav_from_file(yet_another_path)
            #gina.view_spectrogram(yet_another_path)

            source = yet_another_path #"/home/User/Documents/file.txt"
            random_sequence = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            new_f = random_sequence + "_" + f
            destination = join(gina.path_chunks, new_f)
            print(source)
            print(destination)
            gina.add_wav_chunk_to_gina_dictionary_with_label(new_f, dir)
            dest = shutil.copy(source, destination)
            #x = input("press enter")



#print( gina.dictionary_gina.items()  )

gina.write_dictionary_gina() # write to file

######################################################################

print("<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")
