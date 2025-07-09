import os, math
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def process_chunk(chunk, 
                    dff,
                    nn, 
                    quan_95_tanimoto, 
                    df_dmf_fps, 
                    chunk_id=None):

    # Calculate distances and indices for the chunk
    distances_chunk, indices_chunk = nn.kneighbors(chunk)

    # Calculate mean and variance of DGsolv for the 5 nearest neighbors
    nn_dgsolv = df_dmf_fps['DGsolv'].values[indices_chunk.flatten()]
    nn_dgsolv = nn_dgsolv.reshape(distances_chunk.shape)
    nn_dgsolv_mean = nn_dgsolv.mean(axis=1)
    nn_dgsolv_var = nn_dgsolv.var(axis=1)

    print(f'Processing chunk of size {len(dff)}')

    # Create a DataFrame for the chunk results
    chunk_results = pd.DataFrame({
        'can_smiles_solute': dff['can_smiles_solute'],
        'can_smiles_solvent': dff['can_smiles_solvent'],
        'DGsolv': dff['DGsolv'],
        'solubility_predicted': dff['solubility_predicted'],
        'distance_to_train': distances_chunk.mean(axis=1),
        'above_95_quantile_train': (distances_chunk.mean(axis=1) > quan_95_tanimoto).astype(int),
        'DGsolv_5NN': nn_dgsolv_mean,
        'DGsolv_5NN_var': nn_dgsolv_var
    })
    
    # Return the chunk results
    return chunk_results


# main
if __name__ == "__main__":

    # load fingerprints (1 min)
    train_fps = np.load('/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/train_solute_fps.npy')
    test_fps = np.load('/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/test_smiles_fps.npy')

    # deployment data
    dff2fps = pd.read_csv('/home/nanta/Redox_mediator_screening/data/Filtered_data/filter2.csv.gz', compression='gzip')

    # train data
    df_dmf_fps = pd.read_csv('/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/Solv_GNN_SSD_train_DMF_as_solvent.csv')

    # 5-NN Tanimoto distance from train to train
    nn = NearestNeighbors(n_neighbors=5, metric='jaccard')
    nn.fit(train_fps)
    distances, indices = nn.kneighbors(train_fps)
    quan_95_tanimoto = np.quantile(distances, 0.95)
    print(f'95% quantile of Tanimoto distances: {quan_95_tanimoto}')

    # 5-NN Tanimoto distance from test to train
    # create chunk of test data
    chunk_size = 20000  # Adjust based on your memory capacity
    test_chunks = [test_fps[i:i + chunk_size] 
                    for i in range(0, len(test_fps), chunk_size)]  
    dff2_chunks = [dff2fps[i:i + chunk_size] 
                    for i in range(0, len(dff2fps), chunk_size)]

    cc = zip(test_chunks, dff2_chunks)
    
    # ----------- actual multiprocess fan-out
    N_JOBS = min(8, os.cpu_count())   # adjust as you like

    print(f"Launching {len(test_chunks)} workers "
          f"({N_JOBS} running in parallel)…")

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_chunk)(chunk,
                               dff,
                               nn,
                               quan_95_tanimoto,    
                               df_dmf_fps,                               
                               i)
        for i, (chunk, dff) in enumerate(tqdm(cc, desc="Processing test chunks")))

    # Combine results
    combined_results = pd.concat(results, ignore_index=True)

    # Save results
    out_file = '/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/filter2_with_distances_fps.csv.gz'
    combined_results.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

