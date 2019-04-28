from .models import BidirectionalAttentionFlow
from .scripts import load_data_generators
from .scripts import data_download_and_preprocess
import os
# import numpy as np
# from keras.optimizers import Adadelta
# from .scripts import negative_avg_log_error
# from .scripts import accuracy


def main():

    data_download_and_preprocess()
    
    emdim = 400
    bidaf = BidirectionalAttentionFlow(emdim=emdim, num_highway_layers=2,
                                       num_decoders=1, encoder_dropout=0.4, decoder_dropout=0.6)
    # bidaf.load_bidaf(os.path.join(os.path.dirname(__file__), 'saved_items', 'bidaf_07.h5'))
    # adadelta = Adadelta(lr=0.0085)
    # bidaf.model.compile(loss=negative_avg_log_error, optimizer=adadelta, metrics=[accuracy])
    # w1 = bidaf.model.get_weights()
    # bidaf.load_bidaf(os.path.join(os.path.dirname(__file__), 'saved_items', 'bidaf_01.h5'))
    # w2 = bidaf.model.get_weights()
    train_generator, validation_generator = load_data_generators(batch_size=16, emdim=emdim, shuffle=True)
    model = bidaf.train_model(train_generator, epochs=1, validation_generator=validation_generator,
                              save_history=True, save_model_per_epoch=True,steps_per_epoch=5)
    # print("+++++++++++++++++++++++++++++ length", len(w1))
    # for a,b in zip(w1,w2):
    #     if not np.all(a==b):
    #         print("Its working..")
    # scores = bidaf.model.evaluate_generator(train_generator, verbose=1)
    #  print(scores)


if __name__ == '__main__':
    main()