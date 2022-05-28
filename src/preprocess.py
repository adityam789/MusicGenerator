import glob
import os
import pickle
import numpy as np
from keras.utils import to_categorical
from music21 import converter, instrument, note, chord


def parse_midi_files(foldername):
    """
    get all midi files in the folder and parse them to get notes
    foldername: folder name, e.g. './midi_songs/'
    """
    notes = []
    for file in glob.glob(foldername+"/*.mid"):

        midi = converter.parse(file)
        notes_to_parse = None
        print("Parsing %s" % file)

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        
    # print(notes)
    print(len(notes))

    if not os.path.exists("./data"):
        os.makedirs("./data")

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def preprocess_notes(notes, seq_length = 100):
    """
    Convert notes into numeric representation, and transform into a sequencial dataset for model training.
    notes: list of notes 
    seq_length: sequence length (int)
    """
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map notes to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    train_x = []
    train_y = []

    for i in range(0, len(notes) - seq_length, 1):
        train_x.append([note_to_int[note] for note in notes[i:i + seq_length]])
        train_y.append(note_to_int[notes[i + seq_length]])

    train_size = len(train_x)
    
    # Transform the training set into a numpy array and normalize it
    normalized_train_x = np.reshape(train_x, (train_size, seq_length, 1))
    normalized_train_x = normalized_train_x / float(len(notes))

    train_y = to_categorical(train_y)

    return (pitchnames, train_x, normalized_train_x, train_y)