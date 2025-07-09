import os, math
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ————————————————————————————————————————————————————————————
# Load once, keep global.  Each worker will import this module and
# re-initialise TensorFlow; that is unavoidable with multiprocessing.
# ————————————————————————————————————————————————————————————
import tensorflow as tf
import nfp
from gnn import (create_tf_dataset, CustomPreprocessor,
                 atom_features, bond_features, global_features)

MODEL_PATH       = '/home/nanta/Solv_GNN_SSD/model_files/SSD_models/student35/best_model.h5'
PREPROCESSOR_JS  = '/home/nanta/Solv_GNN_SSD/model_files/SSD_models/student35/preprocessor.json'

def init_tf():
    """Make TF behave politely inside every worker."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

    model        = tf.keras.models.load_model(MODEL_PATH,
                                              custom_objects=nfp.custom_objects)
    preprocessor = CustomPreprocessor(explicit_hs=False,
                                      atom_features=atom_features,
                                      bond_features=bond_features)
    preprocessor.from_json(PREPROCESSOR_JS)

    extractor = tf.keras.Model(model.inputs,
                               [model.get_layer('dense_22').input])

    output_signature = (
        preprocessor.output_signature,
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    return extractor, preprocessor, output_signature


# ————————————————————————————————————————————————————————————
# One worker = one chunk
# ————————————————————————————————————————————————————————————
def process_chunk(chunk,
                  emb_dmf,
                  quan_95_dmf,
                  df_dmf_DGsolv,
                  chunk_id=None):

    # local TF objects
    extractor, preprocessor, output_signature = init_tf()

    def embed_dataframe(df):
        ds = tf.data.Dataset.from_generator(
            lambda: create_tf_dataset(df, preprocessor, 1.0, False),
            output_signature=output_signature
        ).padded_batch(batch_size=len(df))
        emb = extractor.predict(ds, verbose=0).squeeze()[:, :128]
        return emb

    # ——— embed the current chunk
    emb_chunk = embed_dataframe(chunk)

    # ——— find neighbours in pre-computed DMF space
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(emb_dmf)
    dist, idx = nn.kneighbors(emb_chunk)

    # ——— stats of ΔGsolv among the neighbours
    nn_dg = df_dmf_DGsolv[idx.ravel()].reshape(dist.shape)
    nn_mean = nn_dg.mean(axis=1)
    nn_var  = nn_dg.var(axis=1)

    # ——— assemble DataFrame
    out = pd.DataFrame({
        "can_smiles_solute":    chunk["can_smiles_solute"],
        "can_smiles_solvent":   chunk["can_smiles_solvent"],
        "DGsolv":               chunk["DGsolv"],
        "solubility_predicted": chunk["solubility_predicted"],
        "distance_to_train":    dist.mean(axis=1),
        "above_95_quantile_train":
                (dist.mean(axis=1) > quan_95_dmf).astype(int),
        "DGsolv_5NN":      nn_mean,
        "DGsolv_5NN_var":  nn_var,
    })

    if chunk_id is not None:
        print(f"[worker] finished chunk {chunk_id:3d} "
              f"({len(chunk):,d} rows)")
    return out


# ————————————————————————————————————————————————————————————
# Main program
# ————————————————————————————————————————————————————————————
if __name__ == "__main__":

    # 1.  reference set (DMF)
    df_dmf = pd.read_csv(
        "/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/"
        "Solv_GNN_SSD_train_DMF_as_solvent.csv"
    )

    extractor0, preproc0, sig0 = init_tf()      # one time only
    def embed_dataframe_once(df):
        ds = tf.data.Dataset.from_generator(
            lambda: create_tf_dataset(df, preproc0, 1.0, False),
            output_signature=sig0).padded_batch(batch_size=len(df))
        return extractor0.predict(ds, verbose=0).squeeze()[:, :128]

    emb_dmf = embed_dataframe_once(df_dmf)

    nn_tmp = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_tmp.fit(emb_dmf)
    dist_tmp, _ = nn_tmp.kneighbors(emb_dmf)
    quan_95_dmf = np.quantile(dist_tmp, 0.95)
    print(f"95 % quantile distance = {quan_95_dmf:.4f}")

    # 2.  target table
    dff2f = (pd.read_csv(
                "/home/nanta/Redox_mediator_screening/data/"
                "Filtered_data/filter2.csv.gz",
                compression="gzip")
              .rename(columns={'predicted': 'solubility_predicted'})
              [['can_smiles_solute', 'can_smiles_solvent',
                'DGsolv', 'solubility_predicted']]
            )

    # 3.  chunk-and-parallel
    CHUNK = 20_000
    chunks = [dff2f.iloc[i:i+CHUNK].copy()
              for i in range(0, len(dff2f), CHUNK)]

    # ----------- actual multiprocess fan-out
    N_JOBS = min(8, os.cpu_count())   # adjust as you like

    print(f"Launching {len(chunks)} workers "
          f"({N_JOBS} running in parallel)…")

    results = Parallel(n_jobs=N_JOBS,
                       backend="loky",
                       batch_size=1,
                       verbose=10)(
        delayed(process_chunk)(chunk,
                               emb_dmf,
                               quan_95_dmf,
                               df_dmf["DGsolv"].values,
                               i)
        for i, chunk in enumerate(chunks))

    dff2f_results = pd.concat(results, ignore_index=True)

    # 4.  save
    out_file = ("/home/nanta/Solv_GNN_SSD/data/DMF_as_solvent/"
                "filter2_with_distances.csv.gz")
    dff2f_results.to_csv(out_file, compression="gzip", index=False)
    print("✓  wrote", out_file)

    # 5.  quick QC
    pct_over = (dff2f_results["above_95_quantile_train"].mean() * 100)
    print(f"{pct_over:5.2f} % of samples lie beyond the 95 % radius.")
