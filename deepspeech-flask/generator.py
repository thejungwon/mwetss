from __future__ import absolute_import, division, print_function
from flask import Flask

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import soundfile
import os
import glob
from deepspeech import Model, printVersions
from timeit import default_timer as timer

import tqdm

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape
    N_pad = N_tar
    # print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append

    if shape[0] > 0:
        if len(shape) > 1:
            front = np.vstack((np.zeros(shape),data))

            return np.vstack((front,np.zeros(shape)))
        else:
            front = np.hstack((np.zeros(shape),data))
            return np.hstack((front, np.zeros(shape)))
    else:
        return data

def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, np.int16)


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        printVersions()
        exit(0)

#deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my_audio_file.wav

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='audio text ground truth generator')
    parser.add_argument('audio_path', help='Path to the input directory.')
    parser.add_argument('total_process', help='Path to the input directory.')
    parser.add_argument('process_number', help='Path to the input directory.')
    return parser

if __name__ == '__main__':
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])



    total_process = int(parsed_args.total_process)
    process_number = int(parsed_args.process_number)
    targets = glob.glob(os.path.join(parsed_args.audio_path, "*target.wav"))
    target_length = len(targets)
    start = int((process_number-1)*target_length/total_process)
    end = int((process_number)*target_length/total_process)


    # print("start",start)
    # print("end",end)
    targets = targets[start:end]
    # print(targets)
    ds = Model("models/output_graph.pbmm", N_FEATURES, N_CONTEXT, "models/alphabet.txt", BEAM_WIDTH)
    ds.enableDecoderWithLM("models/alphabet.txt", "models/lm.binary", "models/trie", LM_ALPHA, LM_BETA)
    #
    for audio_file in tqdm.tqdm(targets):
        file_name = audio_file.split("/")[-1].split("-")[0]
        path = audio_file.split("/")[:-1]
        path = "/".join(path)
        data, samplerate = soundfile.read(audio_file)
        data = pad_audio(data,samplerate,0.5)
        soundfile.write("temp"+str(process_number)+".wav", data, samplerate, subtype='PCM_16')
        fin = wave.open("temp"+str(process_number)+".wav", 'rb')
        fs = fin.getframerate()
        if fs != 16000:
            print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
            fs, audio = convert_samplerate(audio_file)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        audio_length = fin.getnframes() * (1/16000)
        fin.close()

        gt_text = path+"/"+file_name+"-gt.txt"

        file = open(gt_text,"w")#write mode
        file.write(ds.stt(audio, fs))
        file.close()

        # return ds.stt(audio, fs)
