import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint


def model_train(train_x, train_y, n_vocab, recurrent_dropout = 0.3, dropout = 0.3, batch_size = 64, epochs = 200):
    
    model = Sequential()
    model.add(LSTM(512, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(dropout))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(dropout))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if not os.path.exists("./models"):
        os.makedirs("./models")

    filepath = "models/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]     
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    return model, filepath