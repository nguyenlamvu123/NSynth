from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator

# Using embedded configuration.
separator = Separator('spleeter:5stems')
# # Using custom configuration file.
# separator = Separator('/path/to/config.json')

audio_loader = AudioAdapter.default()
sample_rate = 44100
waveform, _ = audio_loader.load('/home/zaibachkhoa/Downloads/bientinh_huonglycover.mp4', sample_rate=sample_rate)

# Perform the separation :
prediction = separator.separate(waveform)
print()