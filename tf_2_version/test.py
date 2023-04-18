import tensorflow as tf
import argparse
import os
import math
import librosa
import soundfile as sf

from model import CycleGAN
from preprocess import *


def loadPickleFile(fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)


def test(model,test_dir,output_dir,mceps):
    os.makedirs(output_dir,exist_ok=True)

    log_files = np.load(os.path.join(mceps, 'logf0s_normalization.npz'))
    log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B = log_files['log_f0s_mean_A'],log_files['log_f0s_std_A'],log_files['log_f0s_mean_B'],log_files['log_f0s_std_B']

    mcep_files = np.load(os.path.join(mceps, 'mcep_normalization.npz'))
    coded_sps_A_mean,coded_sps_A_std,coded_sps_B_mean,coded_sps_B_std = mcep_files['coded_sps_A_mean'],mcep_files['coded_sps_A_std'],mcep_files['coded_sps_B_mean'],mcep_files['coded_sps_B_std']
    
    for file in os.listdir(test_dir):
        filepath = os.path.join(test_dir, file)
        wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
        coded_sp_transposed = coded_sp.T
        coded_sp_norm = np.array((coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std)
        padding = np.zeros((coded_sp_norm.shape[0],n_frames*math.ceil(coded_sp_norm.shape[1]/n_frames) - coded_sp_norm.shape[1]))
        padded_coded_sp_norm = np.concatenate([coded_sp_norm,padding],axis=1)
        preds= []
        for start_frame in range(0,padded_coded_sp_norm.shape[1],n_frames):
            preds.append(np.squeeze(model.test(np.expand_dims(padded_coded_sp_norm[:,start_frame:start_frame+n_frames],axis=0), 'A2B').numpy(),axis=0))
        coded_sp_converted_norm = np.concatenate(preds,axis=1)
        coded_sp_converted = coded_sp_converted_norm[:,:coded_sp_norm.shape[1]] * coded_sps_B_std + coded_sps_B_mean
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
        # librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)
        sf.write(os.path.join(output_dir, os.path.basename(file)), wav_transformed, sampling_rate, 'PCM_24')

    pass

if __name__ == '__main__':
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128
    model = CycleGAN(num_features = num_mcep,num_frames=n_frames)
    
    parser = argparse.ArgumentParser(description = 'Test CycleGAN model for datasets.')

    test_dir_default = './data/vcc2016_training/SF1'
    weight_dir_default = './kaggle/working/epoch19_weights'
    output_dir_default = './validation_output'
    mceps_dir_default = './preprocessed_SF1TM2'

    parser.add_argument('--test_dir', type = str, help = 'Directory for test dataset.', default = test_dir_default)
    parser.add_argument('--weight_dir', type = str, help = 'Directory for saved weights.', default = weight_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for output audio files.', default = output_dir_default)
    parser.add_argument('--mceps_dir', type = str, help = 'Directory for preprocessed data files.', default = mceps_dir_default)
    argv = parser.parse_args()

    weight_dir = argv.weight_dir
    output_dir = argv.output_dir
    test_dir = argv.test_dir
    mcep_dir = argv.mceps_dir
    print("Weight dir",weight_dir)
    model.load(weight_dir)

    test(model,test_dir,output_dir,mcep_dir)
