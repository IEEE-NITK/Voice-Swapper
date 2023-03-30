import os
import tensorflow as tf
from module import Discriminator, Generator
#from tf_2_version.utils import gradient_penalty
from utils import l1_loss, l2_loss, cross_entropy_loss,gradient_penalty
from datetime import datetime
import numpy as np
import glob

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)

class CycleGAN(object):

    def __init__(self, num_features,num_frames=128, batch_size=1, mode = 'train', log_dir = './log',add_noise=False):

        self.num_features = num_features
        self.input_shape = [batch_size, self.num_features, num_frames] # [batch_size, num_features, num_frames]

        self.discriminator = Discriminator().build_discriminator
        self.generator = Generator().generator_gatedcnn
        self.generator_A2B = self.generator(input_shape = self.input_shape , reuse = False, scope_name = 'generator_A2B')
        self.generatorA2B_variables = self.generator_A2B.trainable_weights
        
        self.generator_B2A= self.generator(input_shape = self.input_shape , reuse = False, scope_name = 'generator_B2A')
        self.generatorB2A_variables = self.generator_B2A.trainable_weights
        
        self.disc_A = self.discriminator(input_shape = self.input_shape , reuse = False, scope_name = 'discriminator_A')
        self.disc_B = self.discriminator(input_shape = self.input_shape , reuse = False, scope_name = 'discriminator_B')

        # get disc weights
        self.discA_variables=self.disc_A.trainable_weights
        self.discB_variables=self.disc_B.trainable_weights
        
        self.mode = mode

        self.initialize_placeholders()
        self.optimizer_initializer()
        
        self.add_noise = add_noise

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.create_file_writer(self.log_dir)
            # self.generator_summaries, self.discriminator_summaries = self.summary()

    def initialize_placeholders(self):
        self.cycle_loss=0
        self.identity_loss=0
        self.generator_loss_A2B=0
        self.generator_loss_B2A=0
        self.generator_loss=0
        self.discriminator_loss_A=0
        self.discriminator_loss_B=0
        self.discriminator_loss=0

    @tf.function
    def forward_pass(self, A_real, B_real, lambda_cycle, lambda_identity, generator_learning_rate, discriminator_learning_rate):

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.optimizer_initializer(d_rate=generator_learning_rate,gen_rate=discriminator_learning_rate)


        with tf.GradientTape(persistent=True) as tape:
            # generate signal B from A(real)
            self.B_fake = self.generator_A2B(A_real)
            
            # generate cycle A from B(fake)
            self.cycle_A = self.generator_B2A(self.B_fake)

            # generate signal A from B(real)
            self.A_fake = self.generator_B2A(B_real)
            # generate cycle B from A(fake)
            self.cycle_B = self.generator_A2B(self.A_fake)

            # pass A(real) through generatorB2A for identity-mapping
            self.generation_A_identity = self.generator_B2A(A_real)
            # pass B(real) through generatorA2B for identity-mapping
            self.generation_B_identity = self.generator_A2B(B_real)

            # evaluate fakes with respective discriminators
            self.discrimination_A_fake = self.disc_A(self.A_fake)
            self.discrimination_B_fake = self.disc_B(self.B_fake)

            # Cycle loss
            self.cycle_loss = l1_loss(A_real,self.cycle_A) + l1_loss(B_real,self.cycle_B)
            # Identity loss
            self.identity_loss = l1_loss(A_real,self.generation_A_identity) + l1_loss(B_real,self.generation_B_identity)
            
            # Generator loss
            # Generator wants to fool discriminator
            self.generator_loss_A2B = l2_loss(tf.ones_like(self.discrimination_B_fake),self.discrimination_B_fake)
            self.generator_loss_B2A = l2_loss(tf.ones_like(self.discrimination_A_fake),self.discrimination_A_fake)

            # Merge the two generators and the cycle loss
            self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + self.lambda_cycle * self.cycle_loss + self.lambda_identity * self.identity_loss
            
            # Add noise to the audio data
            if self.add_noise:
                noise = tf.random.normal(shape=tf.shape(A_real), mean=0.0, stddev=0.1)
                A_real_noisy = A_real + noise

                noise = tf.random.normal(shape=tf.shape(B_real), mean=0.0, stddev=0.1)
                B_real_noisy = B_real + noise

                noise = tf.random.normal(shape=tf.shape(self.A_fake), mean=0.0, stddev=0.1)
                A_fake_noisy = self.A_fake + noise

                noise = tf.random.normal(shape=tf.shape(self.B_fake), mean=0.0, stddev=0.1)
                B_fake_noisy = self.B_fake + noise

                # Discriminator loss
                self.discrimination_A_real = self.discriminator(reuse = True, scope_name = 'discriminator_A')(tf.expand_dims(A_real_noisy,-1))
                self.discrimination_B_real = self.discriminator(reuse = True, scope_name = 'discriminator_B')(tf.expand_dims(B_real_noisy,-1))
                self.discrimination_A_fake = self.discriminator(reuse = True, scope_name = 'discriminator_A')(tf.expand_dims(A_fake_noisy,-1))
                self.discrimination_B_fake = self.discriminator(reuse = True, scope_name = 'discriminator_B')(tf.expand_dims(B_fake_noisy,-1))
            else:
                self.discrimination_A_real = self.discriminator(reuse = True, scope_name = 'discriminator_A')(tf.expand_dims(A_real,-1))
                self.discrimination_B_real = self.discriminator(reuse = True, scope_name = 'discriminator_B')(tf.expand_dims(B_real,-1))
                self.discrimination_A_fake = self.discriminator(reuse = True, scope_name = 'discriminator_A')(tf.expand_dims(self.A_fake,-1))
                self.discrimination_B_fake = self.discriminator(reuse = True, scope_name = 'discriminator_B')(tf.expand_dims(self.B_fake,-1))


            # Discriminator wants to classify real and fake correctly
            self.discriminator_loss_input_A_real = l2_loss(y = tf.ones_like(self.discrimination_A_real), y_hat = self.discrimination_A_real)
            self.discriminator_loss_input_A_fake = l2_loss(y = tf.zeros_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)
            self.discriminator_loss_A = (self.discriminator_loss_input_A_real + self.discriminator_loss_input_A_fake) / 2

            self.discriminator_loss_B_real = l2_loss(y = tf.ones_like(self.discrimination_B_real), y_hat = self.discrimination_B_real)
            self.discriminator_loss_B_fake = l2_loss(y = tf.zeros_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)
            self.discriminator_loss_B = (self.discriminator_loss_B_real + self.discriminator_loss_B_fake) / 2

            # Merge the two discriminators into one
            self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B

            # Compute the gradient penalty for domain A
            # grad_penalty_A = gradient_penalty(discriminator_A, real_A, fake_A)
            # Add the gradient penalty to the discriminator loss for domain A
            # loss_D_A += LAMBDA * grad_penalty_A


        # genA2B_grad = gen_tape.gradient(self.generator_loss_A2B,self.generatorA2B_variables)
        # genB2A_grad = gen_tape.gradient(self.generator_loss_B2A,self.generatorB2A_variables)
        genA2B_grad = tape.gradient(self.generator_loss,self.generatorA2B_variables)
        genB2A_grad = tape.gradient(self.generator_loss,self.generatorB2A_variables)

        disc_A_grad = tape.gradient(self.discriminator_loss,self.discA_variables)
        disc_B_grad = tape.gradient(self.discriminator_loss,self.discB_variables)

        # Apply gradients
        # self.generator_optimizer.apply_gradients(zip(genA2B_grad, self.generatorA2B_variables))
        # self.generator_optimizer.apply_gradients(zip(genB2A_grad, self.generatorB2A_variables))
        self.gen_A2B_optimizer.apply_gradients(zip(genA2B_grad, self.generatorA2B_variables))
        self.gen_B2A_optimizer.apply_gradients(zip(genB2A_grad, self.generatorB2A_variables))

        self.disc_A_optimizer.apply_gradients(zip(disc_A_grad, self.discA_variables))
        self.disc_B_optimizer.apply_gradients(zip(disc_B_grad, self.discB_variables))

        # del gen_tape
        # del disc_tape
        del tape

        self.train_step+=1
        return self.generator_loss, self.discriminator_loss

    def optimizer_initializer(self,d_rate=0.0001,gen_rate=0.0002):
        
        if len(tf.config.list_physical_devices('GPU'))>0:
            # self.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate =d_rate, beta_1 = 0.5)
            self.disc_A_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate =d_rate, beta_1 = 0.5)
            self.disc_B_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate =d_rate, beta_1 = 0.5)

            # self.generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = gen_rate, beta_1 = 0.5)
            self.gen_A2B_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = gen_rate, beta_1 = 0.5)
            self.gen_B2A_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = gen_rate, beta_1 = 0.5)
        else:
            # self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate =d_rate, beta_1 = 0.5)
            self.disc_A_optimizer = tf.keras.optimizers.Adam(learning_rate =d_rate, beta_1 = 0.5)
            self.disc_B_optimizer = tf.keras.optimizers.Adam(learning_rate =d_rate, beta_1 = 0.5)

            # self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = gen_rate, beta_1 = 0.5)
            self.gen_A2B_optimizer = tf.keras.optimizers.Adam(learning_rate = gen_rate, beta_1 = 0.5)
            self.gen_B2A_optimizer = tf.keras.optimizers.Adam(learning_rate = gen_rate, beta_1 = 0.5)

    def test(self, inputs, direction):

        if direction == 'A2B':
            # generation = self.sess.run(self.generation_B_test, feed_dict = {self.input_A_test: inputs})
            generation =self.generator_A2B(inputs)
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict = {self.input_B_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation

    def save(self, directory, filename):
        print("Saving Weights...")
        self.generator_A2B.save_weights(f"{directory}/A2B_{filename}_cpkt")
        self.generator_B2A.save_weights(f"{directory}/B2A_{filename}_cpkt")
        
        self.disc_A.save_weights(f"{directory}/disc_A_{filename}_cpkt")
        self.disc_B.save_weights(f"{directory}/disc_B_{filename}_cpkt")

    def load(self, dir):
        print("Loading Weights...")
        a2b,b2a = glob.glob(dir+"/A2B*"), glob.glob(dir+"/B2A*")
        # print("."+a2b[0].split(".")[-2])
        self.generator_A2B.load_weights(a2b[0].split("cpkt")[0]+"cpkt")
        self.generator_B2A.load_weights(b2a[0].split("cpkt")[0]+"cpkt")

        disc_a,disc_b = glob.glob(dir+"/disc_A*"), glob.glob(dir+"/disc_B*")
        self.disc_A.load_weights(disc_a[0].split("cpkt")[0]+"cpkt")
        self.disc_B.load_weights(disc_b[0].split("cpkt")[0]+"cpkt")

    def summary(self):

        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.compat.v1.summary.merge([cycle_loss_summary, identity_loss_summary, generator_loss_A2B_summary, generator_loss_B2A_summary, generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.compat.v1.summary.merge([discriminator_loss_A_summary, discriminator_loss_B_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    BATCH_SIZE,NUM_FEATURES,NUM_FRAMES = 10,24,128
    model = CycleGAN(num_features = NUM_FEATURES,num_frames=NUM_FRAMES)
    print('Graph Compile Successeded.')

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

    # A_real = tf.keras.Input(shape=[24,128],batch_size=10)
    A_real = tf.random.uniform(shape=(BATCH_SIZE,NUM_FEATURES,NUM_FRAMES),dtype=tf.dtypes.float32)
    # B_real = tf.keras.Input(shape=[24,128],batch_size=10)
    B_real = tf.random.uniform(shape=(BATCH_SIZE,NUM_FEATURES,NUM_FRAMES),dtype=tf.dtypes.float32)

    # train loop
    for epoch in range(5):
        print("Epoch: ",epoch)
        gen_loss,disc_loss = model.forward_pass(A_real,B_real,lambda_cycle,lambda_identity,generator_learning_rate,discriminator_learning_rate)
        print("Gen loss",gen_loss,"\nDisc Loss",disc_loss)
        # model.optimize_parameters(gen_tape,disc_tape)
        