#
# run: snakemake --profile profiles/slurm

rule predict_df:
    input:
        "data/DMF_as_solvent/Solv_GNN_SSD_train_DMF_as_solvent.csv"
    output:
        "data/DMF_as_solvent/Solv_GNN_SSD_train_DMF_pred_results.csv.gz",
    conda:
        "envs/tf24gpu.yml"
        # can do "$ snakemake --sdm conda --conda-create-envs-only" in advance to create env...
    params:
        solvent_smiles = "CN(C)C=O", # solvent is DMF
    shell:
       """
        conda info --envs
        python models/Solv_GNN_SSD/main.py -filename {output} -predict_df -modelname SSD_models/student35
        """
