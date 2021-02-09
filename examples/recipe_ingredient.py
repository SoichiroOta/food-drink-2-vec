import json
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir) 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer

from food_drink_2_vec import FoodDrink2Vec


if __name__ == '__main__':
    recipe_data = json.load(open('examples/data/recipe2vec.json'))
    all_ingredients = json.load(open('examples/data/all_ingredients.json'))
    df = pd.DataFrame(recipe_data).sort_index(axis=1) #.drop(all_ingredients, axis=1)

    vocab = list(df['concept']) + all_ingredients
    vector_df = df[all_ingredients].fillna(0.0)
    axis_names = list(vector_df.columns)

    ingredient_vectors = np.eye(len(axis_names))

    fd2v = FoodDrink2Vec(
        vocab=vocab,
        axis_names=axis_names,
        recurrent_activation='tanh',
        lang='ja',
        vs=200000,
    )

    vectors = np.concatenate([vector_df.values, ingredient_vectors])

    scaler = MinMaxScaler((-1, 1))
    scaler.fit(vectors)
    scaled_vectors = scaler.transform(vectors)

    transformer = Normalizer().fit(scaled_vectors)
    unit_vectors = transformer.transform(scaled_vectors)

    fd2v.train(
        vectors=unit_vectors,
        batch_size=16,
        early_stopping=True
    )

    print(fd2v.most_similar(['カレー']))
    print(fd2v.most_similar(positive=['カレー', 'アメリカ'], negative=['日本']))
    print(fd2v.most_similar(positive=['カレー', '中国'], negative=['日本']))

    fd2v.model.save('examples/h5_objects/recipe_ingredient2vec_model2.h5')
    fd2v.kv.save('examples/pkl_objects/recipe_ingredient2vec_kv2.pkl')
    fd2v.preprocessor.save('examples/pkl_objects/recipe_ingredient2vec_preprocessor2.pkl')   