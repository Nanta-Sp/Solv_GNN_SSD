#
# run: snakemake --profile profiles/slurm

rule test:
    output:
        "data/DMF_as_solvent/filter2_with_nn.csv.gz"
    conda:
        "envs/chempy312nb.yml"

    notebook:
        "notebooks/data.py.ipynb"