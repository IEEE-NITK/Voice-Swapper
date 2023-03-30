import os
import numpy as np
import argparse
import time
import librosa
import soundfile as sf
import math
import pickle

from preprocess import *
from model import CycleGAN 
import tensorflow as tf

def loadPickleFile(fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

def train(train_dir=None, model_dir=None, model_name=None, random_seed=None, validation_A_dir=None, validation_B_dir=None, output_dir=None, tensorboard_log_dir=None,model_weights_dir = None,add_noise=False):

    np.random.seed(random_seed)

    num_epochs = 5000
    mini_batch_size = 1 # mini_batch_size = 1 is better
    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 200000
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / 200000
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128
    lambda_cycle = 10
    lambda_identity = 5

    log_files = np.load(os.path.join(train_dir, 'logf0s_normalization.npz'))
    log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B = log_files['log_f0s_mean_A'],log_files['log_f0s_std_A'],log_files['log_f0s_mean_B'],log_files['log_f0s_std_B']

    mcep_files = np.load(os.path.join(train_dir, 'mcep_normalization.npz'))
    coded_sps_A_mean,coded_sps_A_std,coded_sps_B_mean,coded_sps_B_std = mcep_files['coded_sps_A_mean'],mcep_files['coded_sps_A_std'],mcep_files['coded_sps_B_mean'],mcep_files['coded_sps_B_std']
    
    coded_sps_A_norm, coded_sps_B_norm = loadPickleFile(os.path.join(train_dir, 'coded_sps_A_norm.pickle')), loadPickleFile(os.path.join(train_dir, 'coded_sps_B_norm.pickle'))

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)


    model = CycleGAN(num_features = num_mcep,num_frames=n_frames,add_noise=add_noise)
    if model_weights_dir is not None:
        model.load(model_weights_dir)

    for epoch in range(num_epochs):
        
        print('Epoch: %d' % epoch)
        '''
        if epoch > 60:
            lambda_identity = 0
        if epoch > 1250:
            generator_learning_rate = max(0, generator_learning_rate - 0.0000002)
            discriminator_learning_rate = max(0, discriminator_learning_rate - 0.0000001)
        '''

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)

        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):
            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 0:
                lambda_identity = 0
            if num_iterations > 500:
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            generator_loss, discriminator_loss = model.forward_pass(dataset_A[start:end],dataset_B[start:end],lambda_cycle,lambda_identity,generator_learning_rate,discriminator_learning_rate)

            if i % 50 == 0:
                #print('Iteration: %d, Generator Loss : %f, Discriminator Loss : %f' % (num_iterations, generator_loss, discriminator_loss))
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, discriminator_loss))

        # model.save(directory = model_dir, filename = model_name)
        model.save(model_dir,f"{model_name}_epoch{epoch}")

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        if validation_A_dir is not None:
            # if num_iterations % 1000 == 0:
            print('Generating Validation Data B from A...')
            for file in os.listdir(validation_A_dir):
                filepath = os.path.join(validation_A_dir, file)
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
                sf.write(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate, 'PCM_24')
            model.save(model_dir,f"{model_name}_epoch{epoch}")


        if validation_B_dir is not None:
            # if num_iterations % 1000 == 0:
            print('Generating Validation Data A from B...')
            for file in os.listdir(validation_B_dir):
                filepath = os.path.join(validation_B_dir, file)
                wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_B, std_log_src = log_f0s_std_B, mean_log_target = log_f0s_mean_A, std_log_target = log_f0s_std_A)
                coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                coded_sp_transposed = coded_sp.T
                coded_sp_norm = np.array((coded_sp_transposed - coded_sps_B_mean) / coded_sps_B_std)
                padding = np.zeros((coded_sp_norm.shape[0],n_frames*math.ceil(coded_sp_norm.shape[1]/n_frames) - coded_sp_norm.shape[1]))
                padded_coded_sp_norm = np.concatenate([coded_sp_norm,padding],axis=1)
                preds= []
                for start_frame in range(0,padded_coded_sp_norm.shape[1],n_frames):
                    preds.append(np.squeeze(model.test(np.expand_dims(padded_coded_sp_norm[:,start_frame:start_frame+n_frames],axis=0), 'B2A').numpy(),axis=0))
                coded_sp_converted_norm = np.concatenate(preds,axis=1)
                coded_sp_converted = coded_sp_converted_norm[:,:coded_sp_norm.shape[1]] * coded_sps_A_std + coded_sps_A_mean
                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                sf.write(os.path.join(validation_B_output_dir, os.path.basename(file)), wav_transformed, sampling_rate, 'PCM_24')
            model.save(model_dir,f"{model_name}_epoch{epoch}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

    preprocessed_dir_default = './data/vcc2016_training/SF1'
    model_dir_default = './model/sf1_tf2'
    model_name_default = 'sf1_tf2.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/SF1'
    # validation_B_dir_default = './data/evaluation_all/TF2'
    validation_B_dir_default = None
    output_dir_default = './validation_output'
    # output_dir_default = './tf_2_version'
    tensorboard_log_dir_default = './log'
    load_model_weights = 'None'

    parser.add_argument('--preprocessed_dir', type = str, help = 'Directory for preprocessed dataset.', default = preprocessed_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type = str, help = 'Convert validation A after each training epoch. If set none, no conversion would be done during the training.', default = validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type = str, help = 'Convert validation B after each training epoch. If set none, no conversion would be done during the training.', default = validation_B_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)
    parser.add_argument('--load_model',type= str,  help = 'Load weights from this directory', default = load_model_weights)
    parser.add_argument('--add_noise',type= bool,  help = 'Load weights from this directory', default = False)
    argv = parser.parse_args()

    preprocessed_dir = argv.preprocessed_dir
    output_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir
    load_model_weights = None if argv.load_model == 'None' or argv.load_model == 'none' else argv.load_model
    add_noise = argv.add_noise

    train(train_dir = preprocessed_dir, model_dir = output_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir, tensorboard_log_dir = tensorboard_log_dir, model_weights_dir = load_model_weights,add_noise=add_noise)
