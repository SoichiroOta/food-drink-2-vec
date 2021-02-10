import json
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from bpemb import BPEmb
import pandas as pd
import numpy as np

from food_drink_2_vec import KeyedVectors


if __name__ == '__main__':
    recipe_data = json.load(open('examples/data/asahi_recipe2vec.json'))
    df = pd.DataFrame(recipe_data)

    vocab = list(df['concept'])

    bpemb = BPEmb(
        lang='ja',
        vs=200000,
        dim=300
    )
    vectors = np.array([np.mean(bpemb.embed(text), axis=0) for text in vocab])
    print(vectors.shape)

    kv = KeyedVectors(
        vectors,
        vocab
    )

    vector = np.mean(bpemb.embed('ノンアルコール'), axis=0)
    print(kv.similar_by_vector(vector))
    