import pickle

from bpemb import BPEmb
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class FoodDrink2VecPreprocessor:

    def __init__(self, vs=1000000, max_len=None, lang='multi'):
        self.vs = vs
        self.max_len = max_len
        self.dim = 300
        self.bpemb = None
        self.lang = lang

    def _init_bpemb(self):
        self.bpemb = BPEmb(lang=self.lang, vs=self.vs, dim=self.dim)
        return self

    def _set_max_len(self, embs_list):
        self.max_len = np.max([len(embs) for embs in embs_list])
        return self

    def _pad_sequences(self, embs):
        return sequence.pad_sequences(
            embs.T, maxlen=self.max_len
        ).T if embs.shape[0] > 0 else np.zeros((self.max_len, self.dim))

    def make_texts_into_features(self, texts):
        self._init_bpemb()
        embs_list = [self.bpemb.embed(text) for text in texts]
        if self.max_len is None:
            self._set_max_len(embs_list)
        return np.array([self._pad_sequences(embs) for embs in embs_list])

    def save(self, fname):        
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        pickle.load(open(fname))


class FoodDrink2Vec:
    def __init__(self, vocab, axis_names, vs=1000000, max_len=None, hidden_layer_sizes=(8,), activation='tanh', recurrent_activation='hard_sigmoid', dropout_rate=0.0, recurrent_dropout_rate=0.0, preprocessor=None, lang='multi'):
        self.kv = None
        self.model = None
        self.vocab = vocab
        self.axis_names = axis_names
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.vs = vs
        self.max_len = max_len
        self.lang = lang
        if preprocessor:
            self.set_preprocessor(preprocessor)
        else:
            self.init_preprocessor()

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        return self

    def init_preprocessor(self):
        self.preprocessor = FoodDrink2VecPreprocessor(
            vs=self.vs,
            lang=self.lang
        )
        return self

    def _init_model(self, input_shape):
        inputs = Input(shape=input_shape)
        h = Bidirectional(LSTM(
            self.hidden_layer_sizes[0],
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate
        ))(inputs)
        if len(self.hidden_layer_sizes) > 1:
            for hidden_layer_size in self.hidden_layer_sizes[1:]:
                h = Dense(hidden_layer_size, activation=self.activation)(h)
                if self.dropout_rate > 0.0:
                    h = Dropout(self.dropout_rate)(h)
        predictions = Dense(len(self.axis_names), activation='linear')(h)
        model = Model(inputs=inputs, outputs=predictions)
        model.summary()
        self.model = model
        return self

    def dim(self):
        return self.preprocessor.dim

    def preprocess_vocab(self):
        features = self.preprocessor.make_texts_into_features(self.vocab)
        self.max_len = self.preprocessor.max_len
        return features

    def train(self, vectors, epochs=200, optimizer='adam', batch_size=None, early_stopping=False, validation_split=0.01, loss='cosine_proximity', metrics=('mse', 'mae')):
        features = self.preprocess_vocab()
        self._init_model((self.max_len, self.dim()))
        self.model.compile(optimizer, loss, metrics=list(metrics))            
        callbacks = [EarlyStopping()] if early_stopping else []
        self.model.fit(
            features, vectors,
            batch_size=(batch_size if batch_size else 1),
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_split=validation_split
        )

        predicted_vectors = self.model.predict(features)
        self.kv = KeyedVectors(
            predicted_vectors,
            self.vocab,
            self.axis_names
        )
        return self

    def distance(self, text1, text2):
        distances = self.distances(text1, [text2])
        return distances[0]

    def distances(self, text_or_vector, other_texts):
        vector = self.get_vector(text_or_vector) if type(text_or_vector) is str else text_or_vector
        features = self.preprocessor.make_texts_into_features(list(other_texts))
        other_vectors = self.model.predict(features)
        similarity = cosine_similarity(
            np.array([vector]),
            other_vectors
        )
        return 1.0 - similarity[0]


    def get_vector(self, text, norm=False):
        features = self.preprocessor.make_texts_into_features([text])
        predicted_vectors = self.model.predict(features)
        if norm:
            return normalize(predicted_vectors)[0]
        else:
            return predicted_vectors[0]

    def most_similar(self, positive=None, negative=None, topn=10):
        if positive:
            positive_features = self.preprocessor.make_texts_into_features(positive)
            positive_vectors = self.model.predict(positive_features)
            positive_vector = np.sum(normalize(positive_vectors), axis=0)
        else:
            positive_vector = np.zeros(len(self.axis_names))

        if negative:
            negative_features = self.preprocessor.make_texts_into_features(negative)
            negative_vectors = self.model.predict(negative_features)
            negative_vector = np.sum(normalize(negative_vectors), axis=0)
        else:
            negative_vector = np.zeros(len(self.axis_names))

        vector = positive_vector - negative_vector
        return self.kv.similar_by_vector(vector, topn)

    def most_similar_to_given(self, text1, texts_list):
        most_similar_indices = [
            np.argmax([self.similarity(text1, text) for text in texts]) for texts in texts_list
        ]
        return [texts[idx] for texts, idx in zip(texts_list, most_similar_indices)]

    def n_similarity(self, texts1, texts2):
        features1 = self.preprocessor.make_texts_into_features(texts1)
        predicted_vectors1 = self.model.predict(features1)
        features2 = self.preprocessor.make_texts_into_features(texts2)
        predicted_vectors2 = self.model.predict(features2)
        return cosine_similarity(
            predicted_vectors1,
            predicted_vectors2
        )

    def similarity(self, text1, text2):
        n_similarity = self.n_similarity([text1], [text2])
        return n_similarity[0][0]

    @classmethod
    def load_model(cls, fname):
        pickle.load(open(fname))


class KeyedVectors:
    def __init__(self, vectors, vocab, axis_names=None):
        self.index2entity = {idx: entity for idx, entity in enumerate(vocab)}
        self.entity2index = {entity: idx for idx, entity in enumerate(vocab)}
        self.vectors = vectors
        self.axis_names = axis_names if axis_names else list(range(vectors.shape[1]))

    def __getitem__(self, key):
        return self.vectors[self.entity2index[key]]
    
    @classmethod
    def cosine_similarties(cls, vector_1, vectors_all):
        similarties = cosine_similarity(
            np.array([vector_1]),
            np.array(vectors_all)
        )
        return similarties[0]

    def get_normed_vectors(self):
        return normalize(self.vectors)

    def has_index_for(self, key):
        return bool(self.entity2index.get(key))

    def similar_by_key(self, key, topn=10):
        vector = self.vectors[self.entity2index[key]]
        return self.similar_by_vector(vector, topn)

    def similar_by_vector(self, vector, topn=10):
        similarties = KeyedVectors.cosine_similarties(vector, self.vectors)
        sorted_index = np.argsort(similarties)[::-1]
        return [(
            self.index2entity.get(sorted_index[i]),
            similarties[sorted_index[i]]
        ) for i in range(topn)]

    def save(self, fname):        
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        pickle.load(open(fname))