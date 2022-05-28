from src.generate import create_music
from src.model import model_train
from src.predict import func, predict_notes
from src.preprocess import parse_midi_files, preprocess_notes

def main():
    # Parse midi files
    notes = parse_midi_files('CarcassiOP60')

    # get amount of pitch names
    n_vocab = len(set(notes))

    # Preprocess notes
    pitchnames, train_x, normalized_train_x, train_y = preprocess_notes(notes, seq_length = 100)

    print(normalized_train_x.shape)

    # Create the network
    model, model_weight_file_name = model_train(normalized_train_x, train_y, n_vocab, recurrent_dropout = 0.3, dropout = 0.3, batch_size = 128, epochs = 50)

    # Save the model
    # model.save('models/model.h5')

    model_with_weights = func(normalized_train_x, n_vocab, model_weight_file_name, recurrent_dropout = 0.3, dropout = 0.3)

    prediction_output = predict_notes(train_x, model_with_weights, n_vocab, pitchnames, notes_to_predict = 500)

    # Convert the output to midi
    create_music(prediction_output)

    pass

if __name__ == '__main__':
    main()