rule all:
    input:
        # "notebooks/embedding_comparison_out.ipynb"
        'data/DMF_as_solvent/filter2_with_distances.csv.gz',
        'data/DMF_as_solvent/filter2_with_distances_fps.csv.gz'

rule run_embed_comp:
    input:
        # "notebooks/embedding_comparison.ipynb"
        "notebooks/potential_OOD_v2.py"
    output: 
        # "notebooks/embedding_comparison_out.ipynb"
        'data/DMF_as_solvent/filter2_with_distances.csv.gz'
    conda:
        "envs/tf24gpu.yml"
    threads: 8
    shell:
        """
        python {input}
        """
    

# papermill {input} {output}

rule run_fps_comp:
    input:
        "notebooks/potential_OOD_fps.py"
    output: 
        'data/DMF_as_solvent/filter2_with_distances_fps.csv.gz'
    conda:
        "envs/tf24gpu.yml"
    threads: 8
    shell:
        """
        python {input}
        """