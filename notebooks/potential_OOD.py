import pandas as pd
from gnn import create_tf_dataset, CustomPreprocessor, atom_features, bond_features, global_features
from  gnn import *
import nfp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(tf.config.list_physical_devices('GPU'))

if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

df_dmf = pd.read_csv('/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/Solv_GNN_SSD_train_DMF_as_solvent.csv')

# mark the wholde dff2 for in or out of 95% quantile of distances from df_dmf
dff2f = pd.read_csv('/home/nanta/Redox_mediator_screening/data/Filtered_data/filter1_pre_sol_pred_results.csv.gz', compression='gzip')
# dff2f = dff2f.sample(n=10000, random_state=42)
# rename columns for consistency
dff2f.rename(columns={'predicted': 'solubility_predicted'}, inplace=True)
dff2f = dff2f[['can_smiles_solute','can_smiles_solvent','DGsolv','solubility_predicted']]

# Load model and preprocessor
model = tf.keras.models.load_model('/home/nanta/Solv_GNN_SSD/model_files/SSD_models/student35/best_model.h5', custom_objects=nfp.custom_objects)
preprocessor = CustomPreprocessor(explicit_hs=False, atom_features=atom_features, bond_features=bond_features)
preprocessor.from_json('/home/nanta/Solv_GNN_SSD/model_files/SSD_models/student35/preprocessor.json')

extractor = tf.keras.Model(model.inputs, [model.get_layer('dense_22').input]) # layer after all message passing and then tf.concat
output_signature = (preprocessor.output_signature, tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32))

def embed_dataframe(df):
    ds = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(df, preprocessor, 1.0, False),
        output_signature=output_signature
    ).padded_batch(batch_size=len(df))
    # Use tqdm to show progress bar during prediction
    embedding = extractor.predict(ds, verbose=1)  # verbose=1 shows a progress bar
    embedding = embedding.squeeze()
    # select only the first 128 features (solute)
    embedding = embedding[:, :128]
    return pd.DataFrame(embedding, index=df.index)

# embed df_dmf
emb_dmf = embed_dataframe(df_dmf)
nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(emb_dmf)
# calculate distances to df_dmf
distances_dmf, indices_dmf = nn.kneighbors(emb_dmf)
quan_95_dmf = np.quantile(distances_dmf, 0.95)
print(f'95% quantile of distances_dmf: {quan_95_dmf}')

# chunking dff2f to avoid memory issues
chunk_size = 20000  # Adjust based on your memory capacity
dff2f_chunks = [dff2f[i:i + chunk_size] for i in range(0, len(dff2f), chunk_size)]  
# Initialize an empty DataFrame to store results
dff2f_results = pd.DataFrame()  

for chunk in tqdm(dff2f_chunks, desc='Processing chunks'):
    emb_chunk = embed_dataframe(chunk)
    nn_chunk = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_chunk.fit(emb_dmf)
    distances_chunk, indices_chunk = nn_chunk.kneighbors(emb_chunk)
    
    # Calculate mean and variance of DGsolv for the 5 nearest neighbors
    nn_dgsolv = df_dmf['DGsolv'].values[indices_chunk.flatten()]
    nn_dgsolv = nn_dgsolv.reshape(distances_chunk.shape)
    nn_dgsolv_mean = nn_dgsolv.mean(axis=1)
    nn_dgsolv_var = nn_dgsolv.var(axis=1)

    # Create a DataFrame for the chunk results
    chunk_results = pd.DataFrame({
        'can_smiles_solute': chunk['can_smiles_solute'],
        'can_smiles_solvent': chunk['can_smiles_solvent'],
        'DGsolv': chunk['DGsolv'],
        'solubility_predicted': chunk['solubility_predicted'],
        'distance_to_train': distances_chunk.mean(axis=1),
        'above_95_quantile_train': (distances_chunk.mean(axis=1) > quan_95_dmf).astype(int),
        'DGsolv_5NN': nn_dgsolv_mean,
        'DGsolv_5NN_var': nn_dgsolv_var
    })
    
    # Append the chunk results to the main DataFrame
    dff2f_results = pd.concat([dff2f_results, chunk_results], ignore_index=True)
    print(f'Processed chunk of size {len(chunk)}')

# Save the results to a CSV file
dff2f_results.to_csv('/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/filter1_pre_sol_pred_results_with_distances.csv.gz', compression='gzip', index=False)

print(f'Number of samples above 95% quantile in dff2f: {len(dff2f_results[dff2f_results["above_95_quantile_train"]==1])}, percentage: {len(dff2f_results[dff2f_results["above_95_quantile_train"]==1]) / len(dff2f_results) * 100:.2f}%')
print(f'Number of samples below 95% quantile in dff2f: {len(dff2f_results[dff2f_results["above_95_quantile_train"]==0])}, percentage: {len(dff2f_results[dff2f_results["above_95_quantile_train"]==0]) / len(dff2f_results) * 100:.2f}%')