## Gina Module - Speech Intelligent Agent
## www.rcalix.com
## Ricardo A. Calix, Ph.D.
#####################################################################################

import pyaudio
import argparse
import tempfile
import queue
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
import wave
import getopt
import alsaaudio
import pydub
from pydub import AudioSegment
import audiosegment  ## different from previous
from pydub.silence import split_on_silence
import pickle
import os
from os import listdir
from os.path import isfile, join
import shutil
import scipy
import scipy.io.wavfile
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
import random
import string
from scipy.spatial import distance
import librosa   #for audio processing
from scipy.io import wavfile #for audio processing

##################################################################################

from GinaText import GinaText
ginaText = GinaText()

##################################################################################
## key  in dictionary_gina is speech file name id which is unique
## dictionary_gina[filepath][text] = word
## dictionary_gina[filepath][feature_vector] = feature_vector

##################################################################################

class GinaSpeech():

    def __init__(self):
        self.dictionary_gina = {}
        self.dictionary_file_name = "dictionary/dictionary_gina.txt"
        self.dictionary_phonemes = {}
        self.dictionary_phonemes_file_name = "dictionary/dictionary_phonemes.txt"
        self.channel = 1 #test with 2
        self.subtype = "PCM_24"
        self.device_mic = 0  ## microphone
        self.samplerate = 44100
        self.device_speaker = 'default'
        self.path_chunks = "audio/chunks/"
        self.path_utterances = "audio/utterances/"
        self.path_phonemes = "audio/phonemes/"
        self.min_silence_len = 250  ## 0.25 of a second
        self.keep_silence = 250  ## 0.25 of a second
        self.phoneme_length = 200 ## 200 miliseconds ## 2/10 of a second
        ## 25 * 1000 ##setting minimum length of each chunk to 25 seconds
        self.target_length = 1 * 1000  ## 1 second
        self.utterance_prefix = 'audio_file_'
        self.chunk_prefix = 'audio_chunk_'
        self.phoneme_prefix = 'audio_phoneme_'
        self.suffix = '.wav'
        self.initialization_text = "zxcvbnm"
        self.initialization_vector_value = "vector"
        self.q = queue.Queue()
        self.vector_space_phonemes = []
        self.spectrogram_dimensions = (662, 12)
        self.select_top_n_matching_phonemes = 3
        
    #################################################################################
    ## This is called (from a separate thread) for each audio block.

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(  indata.copy()  )

    ###############################################################################
    ## soundfile expects an int, sounddevice provides a float
    ## device_info['default_samplerate'] = 44100

    def record_wav_utterance(self):
        try:
            device_info = sd.query_devices(self.device_mic, 'input')
            samplerate = int(device_info['default_samplerate'])
            new_name_file = tempfile.mktemp(prefix=self.utterance_prefix, suffix=self.suffix, dir='')
            fname = join(self.path_utterances, new_name_file)
            with sf.SoundFile(fname, mode='x', samplerate=samplerate, channels=self.channel, subtype=self.subtype) as file:
                with sd.InputStream(samplerate=samplerate, device=self.device_mic, channels=self.channel, callback=self.callback):
                    print('**************************************')
                    print('press Ctrl+C to stop the recording')
                    print('**************************************')
                    while True:
                        file.write(self.q.get())

        except KeyboardInterrupt:
            print('\nRecording finished: ' + repr(fname))
            return new_name_file
        except Exception as e:
            print("error - RC")

    ###########################################################################################

    def write_chunk_wav(self, chunk):
        new_name_file = tempfile.mktemp(prefix=self.chunk_prefix, suffix='.wav', dir='')
        fname = join(self.path_chunks, new_name_file)
        chunk.export(fname, format="wav")
        return new_name_file

    ###########################################################################################

    def write_phoneme_wav(self, phoneme):
        temp_name = tempfile.mktemp(prefix=self.phoneme_prefix, suffix='.wav', dir='')
        random_sequence = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        new_f = random_sequence + "_" + temp_name
        fname = join(self.path_phonemes, new_f)
        phoneme.export(fname, format="wav")
        return new_f

    ###########################################################################################

    def play(self, device, f):
        print('%d channels, %d sampling rate\n' % (f.getnchannels(), f.getframerate()))
        device.setchannels(f.getnchannels())
        device.setrate(f.getframerate())
        ## 8bit is unsigned in wav files
        if f.getsampwidth() == 1:
            device.setformat(alsaaudio.PCM_FORMAT_U8)
        # Otherwise we assume signed data, little endian
        elif f.getsampwidth() == 2:
            device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        elif f.getsampwidth() == 3:
            device.setformat(alsaaudio.PCM_FORMAT_S24_3LE)
        elif f.getsampwidth() == 4:
            device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
        else:
            raise ValueError('Unsupported format')

        periodsize = f.getframerate() // 8
        device.setperiodsize(periodsize)
        data = f.readframes(periodsize)
        while data:
            # Read data from stdin
            device.write(data)
            data = f.readframes(periodsize)
            # print(len(data))

    #############################################################################

    def play_wav_from_file(self, wav_to_read):
        f = wave.open(wav_to_read, 'rb')
        device_object = alsaaudio.PCM(device=self.device_speaker)
        self.play(device_object, f)
        f.close()
        device_object.close()

    #################################################################

    def split_wav_to_chunks(self, filepath):
        sound = AudioSegment.from_wav(filepath)
        dBFS = sound.dBFS
        chunks = split_on_silence(sound,
                                  min_silence_len=self.min_silence_len,
                                  silence_thresh=dBFS - 16,
                                  keep_silence=self.keep_silence
                                  )
        target_length = self.target_length
        output_chunks = [chunks[0]]
        for chunk in chunks[1:]:
            if len(output_chunks[-1]) < target_length:
                output_chunks[-1] += chunk
            else:
                # if the last output chunk is longer than the target length,
                # we can start a new one
                output_chunks.append(chunk)
        list_chunk_file_names = []
        for chunk in output_chunks:
            chunk_path_name = self.write_chunk_wav(chunk)
            list_chunk_file_names.append(chunk_path_name)
        return list_chunk_file_names

    ######################################################################

    def add_wav_chunk_to_gina_dictionary(self, chunk_path_name):
        self.dictionary_gina[chunk_path_name] = {}
        self.dictionary_gina[chunk_path_name]['text'] = self.initialization_text
        self.dictionary_gina[chunk_path_name]['features'] = self.initialization_vector_value

    ######################################################################

    def add_wav_chunk_to_gina_dictionary_with_label(self, chunk_path_name, label):
        self.dictionary_gina[chunk_path_name] = {}
        self.dictionary_gina[chunk_path_name]['text'] = label
        self.dictionary_gina[chunk_path_name]['features'] = self.initialization_vector_value

    ######################################################################

    def add_phoneme_to_dictionary(self, phoneme, letter, features):
        self.dictionary_phonemes[phoneme] = {}
        self.dictionary_phonemes[phoneme]['letter'] = letter
        self.dictionary_phonemes[phoneme]['features'] = features

    ######################################################################

    def load_dictionary_gina(self):
        with open(self.dictionary_file_name, 'rb') as handle:
            b = pickle.loads(handle.read())
            self.dictionary_gina = b
            
    ######################################################################
            
    def load_dictionary_phonemes(self):
        with open(self.dictionary_phonemes_file_name, 'rb') as phonemes_handle:
            c = pickle.loads(phonemes_handle.read())
            self.dictionary_phonemes = c

    ######################################################################

    def write_dictionary_gina(self):
        with open(self.dictionary_file_name, 'wb') as handle:
            pickle.dump(self.dictionary_gina, handle)
            
    ######################################################################
    
    def write_dictionary_phonemes(self):
        with open(self.dictionary_phonemes_file_name, 'wb') as phonemes_handle:
            pickle.dump(self.dictionary_phonemes, phonemes_handle)

    ######################################################################

    def annotate_speech_chunk_files(self):
        print("****************************************")
        print("begin annotation process...  ")
        keys_to_annotate = [k for (k, v) in self.dictionary_gina.items() if v['text'] == self.initialization_text]
        for key in keys_to_annotate:
            self.play_wav_from_file(  join(self.path_chunks,key)  )
            x = input("enter word   ")
            self.dictionary_gina[key]['text'] = x

    ######################################################################

    def play_wav_chunk_from_text(self, text):
        print(text)
        listOfKeys = [key for (key, value) in self.dictionary_gina.items() if value['text'] == text]
        for temp_fname in listOfKeys:
            self.play_wav_from_file(join(self.path_chunks, temp_fname))

    ######################################################################

    def view_spectrogram(self, wavfile):
        fs, wave = scipy.io.wavfile.read(wavfile)
        spec = plt.specgram(wave, NFFT=int(fs*0.005), Fs=fs, cmap=plt.cm.gray_r, pad_to=256, noverlap=int(fs*0.0025))
        plt.show()

    #####################################################################

    def get_features_spectrogram_audiosegment(self, wavfile):
        seg = audiosegment.from_file(wavfile)
        freqs, times, amplitudes = seg.spectrogram(window_length_s=0.03, overlap=0.5)
        amplitudes = 10 * np.log10(amplitudes + 1e-9)
        return (freqs, times, amplitudes)

    #####################################################################

    def view_spectrogram_audiosegment(self, wavfile):
        seg = audiosegment.from_file(wavfile)
        freqs, times, amplitudes = seg.spectrogram(window_length_s=0.03, overlap=0.5)
        amplitudes = 10 * np.log10(amplitudes + 1e-9)
        # Plot
        plt.pcolormesh(times, freqs, amplitudes)
        plt.xlabel("Time in Seconds")
        plt.ylabel("Frequency in Hz")
        plt.show()
        plt.pcolormesh(amplitudes)
        plt.xlabel("Time in Seconds")
        plt.ylabel("Frequency in Hz")
        plt.show()

    ######################################################################

    def view_plotwave(self, wavfile, maxf=None):
        """Visualize (a segment of) a wave file."""
        # maxf = maximum number of frames
        fs, signal = scipy.io.wavfile.read(wavfile)
        frames = scipy.arange(signal.size)  # x-axis
        if maxf:
            plt.plot(frames[:maxf], signal[:maxf])
            plt.xticks(scipy.arange(0, maxf, 0.5 * fs), scipy.arange(0, maxf / fs, 0.5))
            plt.show()
        else:
            plt.plot(frames, signal)
            plt.xticks(scipy.arange(0, signal.size, 0.5 * fs), scipy.arange(0, signal.size / fs, 0.5))
            plt.show()

    ######################################################################

    def view_fftplot(self, wavfile):
        fs, signal = scipy.io.wavfile.read(wavfile)
        size = signal.size
        fftresult = abs(scipy.fft(signal) / size)
        freqs = scipy.arange(size) * fs / size
        halfsize = int(size / 2)
        plt.plot(freqs[:halfsize], fftresult[:halfsize])
        plt.show()
        plt.plot(freqs, fftresult)
        plt.show()

    ######################################################################

    def view_stats_wave_file(self, wavfile):
        fs, wave = scipy.io.wavfile.read(wavfile)
        print('Data: ', wave)
        print('Sampling rate: ', fs)
        print('Audio length: ', wave.size / fs, ' seconds')
        print('Lowest amplitude: ', min(wave))
        print('Average amplitude: ', sum(wave)/len(wave))
        print('Highest amplitude: ', max(wave))

    ######################################################################

    def convert_wav_chunk_to_feature_vector(self, chunk):
        feature_vector = {}
        return feature_vector

    ########################################################################
    
    def convert_wav_chunks_to_feature_vectors(self):
        list_to_extract = [k for k, v in self.dictionary_gina.items() if v['features'] == self.initialization_vector_value]
        if list_to_extract:
            print(list_to_extract)

        for chunk in list_to_extract:
            try:
                feature_vector = {}
                print(chunk)
                #self.play_wav_from_file(chunk)
                #feature_vector = self.convert_wav_chunk_to_feature_vector(chunk)
                self.dictionary_gina[chunk]['features'] = feature_vector
            except:
                print("error converting wav chunk to feature vector")

    ########################################################################

    def add_new_wav_chunks_to_dictionary(self):
        list_to_extract=[f for f in listdir(self.path_chunks) if isfile(join(self.path_chunks,f))]
        chunks_already_in_gina_dict = [k for k, v in self.dictionary_gina.items()]
        main_list = np.setdiff1d( list_to_extract, chunks_already_in_gina_dict )
        # yields the elements in list_to_extract that are NOT in chunks_already_in_gina_dict
        for chunk in main_list:
            self.dictionary_gina[chunk] = {}
            self.dictionary_gina[chunk]['text'] = self.initialization_text
            self.dictionary_gina[chunk]['features'] = self.initialization_vector_value

    #######################################################################
    
    def split_chunk_into_phonemes(self, filepath):
        sound = audiosegment.from_file(filepath)
        sound = sound.filter_silence()
        ## list of tuples (wav, timestamp) ## phonemes_list
        phonemes_list = sound.generate_frames_as_segments(self.phoneme_length, zero_pad=True)

        list_phoneme_file_names = []
        for phoneme, timestamp in phonemes_list:
            phoneme_path_name = self.write_phoneme_wav(phoneme)
            list_phoneme_file_names.append(phoneme_path_name)
        return list_phoneme_file_names

    ######################################################################

    def build_vector_space_phonemes(self):
        list_of_letters = []
        list_of_vectors = []
        for k, v in self.dictionary_phonemes.items():
            if self.spectrogram_dimensions == v["features"][2].shape and v["letter"] != "":
                #print(k)
                #print(v["letter"])
                #print(v["features"][2].flatten() )
                #print(v["features"][2].shape)
                #x = input()
                list_of_letters.append(   v["letter"]                   )
                list_of_vectors.append(   v["features"][2].flatten()    )

        self.vector_space_phonemes = (  list_of_letters   ,  list_of_vectors  )

    ######################################################################

    def predict_letter_from_phoneme(self, features):
        vector = features[2].flatten()
        vector = np.array(vector).reshape(1, -1)
        #print(vector.shape)
        matrix = np.array(self.vector_space_phonemes[1])
        #print(matrix.shape)
        letters = self.vector_space_phonemes[0]
        #print(len(letters))
        similarities = distance.cdist(matrix, vector, 'euclidean') ## 'cosine'
        #print(similarities)
        indeces = similarities.argsort(0) # axis 0 - the rows
        #print(indeces)
        selected = indeces[:self.select_top_n_matching_phonemes] ## this is still 2d numpy array
        selected = selected.flatten().tolist()
        #print(selected)
        result = [letters[index] for index in selected]
        return result


    ######################################################################
    
    def predict_text_from_chunk(self, filepath):
        list_phonemes = self.split_chunk_into_phonemes(filepath)
        list_of_letters = []
        for phoneme in list_phonemes:
            print(phoneme)
            self.play_wav_from_file(  join(self.path_phonemes, phoneme)   )
            ##self.view_spectrogram_audiosegment(join(self.path_phonemes, phoneme)) # this is cool
            features = self.get_features_spectrogram_audiosegment(join(self.path_phonemes, phoneme) )
            letters_predicted = self.predict_letter_from_phoneme(features)
            for letter_predicted in letters_predicted:
                list_of_letters.append(letter_predicted)
            print("predicted letter: ", list_of_letters)
            x = "000"
            x = input("what is the more likely letter? type it  ")
            self.add_phoneme_to_dictionary(phoneme, x, features)
            list_of_letters = ginaText.resolve_predicted_text(list_of_letters)
        return list_of_letters
    
    #######################################################################

  
