import sys
import pandas as pd


def main():
    if len(sys.argv) < 2:
        # print("Usage: Filter2_solubility.py inputfile sol_thres")
        sys.exit(1)

    # sys.argv[1]: inputfile
    # sys.argv[4]: solubility threshold
    
    df = pd.read_csv(sys.argv[1], compression='gzip')
    df.rename(columns={'predicted': 'solubility_predicted'}, inplace=True)

    threshold = float(sys.argv[4])
    df = df[df['solubility_predicted']<threshold]
    print("After filter2: ", df.shape)

    df.to_csv(sys.argv[2], compression='gzip')
    df['can_smiles_solute'].to_csv(sys.argv[3], index=False, header=False)

    chunk_size = 10000
    for i in range(0, len(df), chunk_size):
    # for i in range(0, 12000, chunk_size):
        chunk = df['can_smiles_solute'][i:i + chunk_size]
        chunk.to_csv(f"{sys.argv[3].split('.')[0]}_part{i//chunk_size + 1}.smi", index=False, header=False)

    with open(f"{sys.argv[3].split('.')[0]}_parts_list.csv", 'w') as parts_file:
        for i in range(0, len(df), chunk_size):
            parts_file.write(f"part{i//chunk_size + 1}\n")

    return 0


if __name__ == "__main__":
    main()