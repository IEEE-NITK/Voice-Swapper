import librosa
import soundfile as sf
import argparse
import os

def crop(path,output_dir,crop_length):
    # Load the WAV file
    audio, sr = librosa.load(path)

    # Crop the first x minutes
    audio_crop = audio[:crop_length*60*sr]

    # Save the cropped audio to a new file
    sf.write(f'audio_cropped.wav', audio_crop, sr, 'PCM_24')


def divide_audio(path,segment_duration=4):     # Define the duration of each segment (in seconds)
    # Load the audio file
    audio, sr = librosa.load(path, sr=None)

    # Calculate the number of samples per segment
    segment_samples = int(segment_duration * sr)

    # Calculate the total number of segments
    total_segments = int(len(audio) / segment_samples)

    os.makedirs('./segmented_files',exist_ok=True)
    # Iterate over the segments and save each segment as a separate audio file
    for i in range(total_segments):
        start = i * segment_samples
        end = (i+1) * segment_samples
        segment = audio[start:end]
        filename = f'./segmented_files/segment_{329+i}.wav'
        sf.write(filename, segment, sr)


if __name__ == "__main__":
    CROP_LENGTH_DEFAULT = 8 #in minutes

    parser = argparse.ArgumentParser(description = 'Crop wav file to a certain size .')
    parser.add_argument('--path', type = str, help = 'Directory for preprocessed dataset.')
    parser.add_argument('--output_dir', type = str, help = 'Directory for output dataset.')
    parser.add_argument('--crop_length', type = str, help = 'Directory for preprocessed dataset.', default = CROP_LENGTH_DEFAULT)
    argv = parser.parse_args()

    path = argv.path
    crop_length = int(argv.crop_length)
    output_dir = argv.output_dir

    # crop(path,output_dir,crop_length)
    divide_audio(path)
    print(
    "Finished Writing!"
    )
