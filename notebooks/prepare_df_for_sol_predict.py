
import sys
import pandas as pd

def prepare_df_for_sol_predict(inputfile, outputfile, solvent_smiles):

    print(inputfile, outputfile, solvent_smiles)
    df = pd.read_csv(inputfile, compression='gzip')

    df['can_smiles_solute'] = df['smiles']
    df['can_smiles_solvent'] = str(solvent_smiles)    
    df['DGsolv'] = ""

    df.to_csv(outputfile, compression='gzip')

    return 0


def main():
    if len(sys.argv) < 3:
        # print("Usage: python prepare_df_for_sol_predict.py inputfile outputfile solvent_smiles ")
        sys.exit(1)
    
    print(sys.argv[1], sys.argv[2], sys.argv[3])

    prepare_df_for_sol_predict(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == "__main__":
    main()