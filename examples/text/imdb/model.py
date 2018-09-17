import pickle
import argparse
import os
import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.layers import (Embedding, Dropout, Conv1D, GlobalMaxPooling1D, 
                                     Dense, Activation)
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ImdbClassifier(object): 

    def __init__(self, max_sequence_length, vocab_size, **kwargs): 
        self._max_sequence_length = max_sequence_length
        self._vocab_size = vocab_size

        self._embedding_size = kwargs.pop('embedding_size', 50)
        self._filters = kwargs.pop('filters', 64)
        self._kernel_size = kwargs.pop('kernel_size', 3)
        self._hidden_dims = kwargs.pop('hidden_dims', 250)

        self._model = None

    def build_model(self): 
        inputs = Input(shape=(self._max_sequence_length,), dtype='int32', name='inputs')

        embed = Embedding(self._vocab_size, self._embedding_size)(inputs)
        conv_1d = Conv1D(self._filters, self._kernel_size, 
                         padding='same', activation='relu', strides=1)(embed)
        max_pool = GlobalMaxPooling1D()(conv_1d)
        dense = Dense(self._hidden_dims, activation='relu')(max_pool)
        predictions = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=inputs, outputs=predictions)

        return model

    @property
    def model(self): 
        if not self._model: 
            self._model = self.build_model()
        return self._model

    def save(self, path): 
        self.model.save(path)

    def load_weights(self, weights_path): 
        self.model.load_weights(weights_path, by_name=True)

    def fit(self, X_train, y_train, X_test, y_test, **kwargs): 
        epochs = kwargs.pop('epochs', 1)
        batch_size = kwargs.pop('batch_size', 64)
        model_path = kwargs.pop('model_path', '/tmp/imdb_cnn')

        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', 
                                                                    patience=3, 
                                                                    min_lr=0.00001, 
                                                                    verbose=1)
        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True)

        self.model.compile(optimizer=Adam(lr=0.01), 
                           loss='binary_crossentropy', 
                           metrics=['accuracy'])

        print(self.model.summary())

        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_test, y_test), 
                                 batch_size=batch_size, 
                                 epochs=epochs, 
                                 callbacks=[early_stopping, reduce_learning_rate, model_checkpoint])
        return history

    def predict(self, X, **kwargs):
        batch_size = kwargs.pop('batch_size', 64)

        return self.model.predict(X, batch_size=batch_size)


def load_data(max_sequence_length, max_features): 
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features, index_from=3)

    stoi = imdb.get_word_index()
    stoi ={k:(v+3) for k,v in stoi.items()}
    stoi["<PAD>"] = 0
    stoi["<START>"] = 1
    stoi["<UNK>"] = 2

    itos = {value:key for key,value in stoi.items()}

    X_train = pad_sequences(X_train, maxlen=max_sequence_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_sequence_length, padding='post')

    return X_train, y_train, X_test, y_test, itos


def train(args): 
    if not os.path.exists(args.output_path): 
        os.makedirs(args.output_path)

    X_train, y_train, X_test, y_test, itos = load_data(args.max_seq_len, args.max_features)

    clf = ImdbClassifier(args.max_seq_len, args.max_features, 
                         embedding_size=args.embedding_size,
                         hidden_dims=args.hidden_dims)

    clf.fit(X_train, y_train, X_test, y_test, 
            batch_size=args.batch_size, 
            epochs=args.epochs,
            model_path=os.path.join(args.output_path, 'imdb_cnn.h5'))


def generate_predictions(args): 
    X_train, y_train, X_test, y_test, itos = load_data(args.max_seq_len, args.max_features)

    clf = ImdbClassifier(args.max_seq_len, args.max_features, 
                         embedding_size=args.embedding_size,
                         hidden_dims=args.hidden_dims)
    clf.load_weights(os.path.join(args.output_path, 'imdb_cnn.h5'))

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    np.save(os.path.join(args.output_path, 'pred_train.npy'), pred_train)
    np.save(os.path.join(args.output_path, 'pred_test.npy'), pred_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMDB classifer') 

    parser.add_argument('mode', type=str, help='train or prediction')
    parser.add_argument('--output-path', dest='output_path', type=str,
                        help='Path to save the model')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--embedding-size', dest='embedding_size', type=int, default=50)
    parser.add_argument('--hidden-dims', dest='hidden_dims', type=int, default=250)
    parser.add_argument('--max-seq-len', dest='max_seq_len', type=int, default=500)
    parser.add_argument('--max-features', dest='max_features', type=int, default=20000)

    args = parser.parse_args()
    if args.mode == 'train': 
        train(args)
    elif args.mode == 'prediction':
        generate_predictions(args)
    else: 
        raise ValueError('Unknown mode {}'.format(args.mode))


