import tensorflow as tf

from config import CONFIG
from database import *
from cae import *
from vae import *

if __name__ == '__main__':

    for latent in [2, 10, 20]:
        for input_len in [100, 500]:
            g = tf.Graph()
            with g.as_default() as g:
                print("STARTED", latent, input_len)
                database = Database("formatted_data/interpolation_" + str(input_len), CONFIG.BATCH_SIZE, input_len,
                                    CONFIG.NUM_CLASSES)
                nn = CAE(database, latent)
                nn.create_model()
                nn.train(evaluate_every=CONFIG.EVALUATE_EVERY, periodic_save=CONFIG.PERIODIC_SAVE,
                         save_location="save/cae_" + str(latent) + "_" + str(input_len) + "/checkpoint.ckpt", max_iter=CONFIG.MAX_ITER)
                nn.close()

    for latent in [2, 10, 20]:
        for input_len in [100, 500]:
            g = tf.Graph()
            with g.as_default() as g:
                print("STARTED", latent, input_len)
                database = Database("formatted_data/interpolation_" + str(input_len), CONFIG.BATCH_SIZE, input_len,
                                    CONFIG.NUM_CLASSES)
                nn = VAE(database, latent)
                nn.create_model()
                nn.train(evaluate_every=CONFIG.EVALUATE_EVERY, periodic_save=CONFIG.PERIODIC_SAVE,
                         save_location="save/vae_" + str(latent) + "_" + str(input_len) + "/checkpoint.ckpt",
                         max_iter=CONFIG.MAX_ITER)
                nn.close()