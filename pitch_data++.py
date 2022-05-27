#!pip install pydub

from pydub import AudioSegment
import IPython
import glob

path = '/content/'
split = 'train/'

audios = glob.glob(path + split + '*.wav')

output_dir = '/content/drive/MyDrive/vocalisation/train_augmented_0.45/'
octaves = -0.45   # -0.4, -0.45, -0.5

for filename in audios: 
  sound = AudioSegment.from_file(filename, format=filename[-3:])
  new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
  hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
  hipitch_sound = hipitch_sound.set_frame_rate(44100)
  new_name = f'aug_{octaves}_' + filename[filename.rfind('/')+1:]
  hipitch_sound.export(f"{output_dir}{new_name}", format="wav")


IPython.display.Audio(filename)
IPython.display.Audio('/content/octave_-0.40.wav')
IPython.display.Audio('/content/octave_-0.45.wav')
IPython.display.Audio('/content/octave_-0.50.wav')