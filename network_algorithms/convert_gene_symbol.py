import pandas as pd

df_hm = pd.read_csv('./lr_human_to_mouse.csv')
df_hm = df_hm.drop_duplicates(subset=['hgnc_symbol'], keep=False).reset_index(drop=True)[['hgnc_symbol', 'external_gene_name']]
print(len(df_hm), len(df_hm['hgnc_symbol'].unique()))

df_novel = pd.read_csv('../st_filtering/data/lr_novel.csv')
df_novel = df_novel.merge(df_hm, left_on=['ligand'], right_on=['hgnc_symbol']).drop(columns=['hgnc_symbol']).rename(columns={'external_gene_name': 'ligand_m'})
df_novel = df_novel.merge(df_hm, left_on=['receptor'], right_on=['hgnc_symbol']).drop(columns=['hgnc_symbol']).rename(columns={'external_gene_name': 'receptor_m'})
df_novel = df_novel[['ligand_m', 'receptor_m', 'type']].rename(columns={'ligand_m':'ligand', 'receptor_m':'receptor'})
df_novel['species'] = 'Mouse'
print(df_novel)
df_novel.to_csv('../st_filtering/data/lr_novel_mouse.csv', index=False)
