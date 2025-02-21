import pandas as pd
import numpy as np

df = pd.read_csv('PubChem_compound_cache_Filter3_THF_0_results.csv')

for i in range(10000,210000,10000):
    print(i)
    temp = pd.read_csv(f'PubChem_compound_cache_Filter3_THF_{i}_results.csv')
    df = pd.concat([df,temp],axis=0)

df.to_csv('PubChem_compound_cache_Filter3_THF_results_all.csv',index=False)