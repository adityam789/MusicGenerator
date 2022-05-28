import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization as BatchNorm


def func(train_x, n_vocab, model_weight_file_name, recurrent_dropout = 0.3, dropout = 0.3):

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

    # Load the weights to each node
    model.load_weights(model_weight_file_name+'.hdf5')

    return model

def predict_notes(train_x, model, n_vocab, pitchnames, notes_to_predict = 500):

    start = np.random.randint(0, len(train_x)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = train_x[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(notes_to_predict):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output