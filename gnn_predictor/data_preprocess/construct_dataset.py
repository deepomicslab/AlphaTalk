import pandas as pd
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split


random.seed(42); np.random.seed(42)


def get_negative_receptor(ligand_gene, positive_genes, df_pool, fold=1):
    """
    :param ligand_gene: str
    :param positive_genes: list of positive receptor gene names
    :param df_pool: pandas.Dataframe, potential ligands and receptors
    :param fold: # of negative/# of positive = fold
    :return: list of negative receptor gene names
    """
    all_genes = [ligand_gene] + positive_genes
    removed_uniprot = df_pool.query('gene in @all_genes')['uniprot'].unique()
    # select receptor + remove proteins
    df_pool = df_pool.query('lr_type == "receptor" and uniprot not in @removed_uniprot').reset_index(drop=True)
    df_pool = df_pool.sample(n=len(positive_genes) * fold, replace=False, axis=0)
    negative_receptor = df_pool[['gene', 'uniprot']].values

    return negative_receptor


def split_data(df, val_size, test_size):
    """
    Separate data according ligand genes, i.e., the ligand genes in test_set are not seen during the training.
    """
    ligand_genes = df['ligand'].unique().tolist()
    np.random.seed(42)
    np.random.shuffle(ligand_genes)
    test_num = round(len(ligand_genes) * test_size)
    train_genes = ligand_genes[test_num:]
    test_genes = ligand_genes[:test_num]
    train_data = df.query('ligand in @train_genes').reset_index(drop=True)
    test_data = df.query('ligand in @test_genes').reset_index(drop=True)
    train_data['set'] = 'train'
    test_data['set'] = 'test'

    train_data, val_data = train_test_split(train_data, test_size=val_size)
    val_data['set'] = 'val'

    out_df = pd.concat((train_data, val_data, test_data)).reset_index(drop=True)
    return out_df


if __name__ == '__main__':
    """
    Use LRDB as positive samples to curate a dataset with positive and negative samples, 
    and split the dataset to training/validation/test sets.
    """

    df_dir = r'../data/df'
    species = "Human"
    negative_fold = 1
    val_size = 300
    test_size = 0.2

    """
    ############################ combine df_lr_uniprot and df_lr ############################
    df_lr_uniprot (only human) is generated from df_af_F1 and uniprot.org
    Each gene has only one corresponding uniprot id and the uniprot is also in df_af_F1['uniprot'].
    """
    df_lr_uniprot = pd.read_csv(os.path.join(df_dir, 'df_lr_uniprot_final.csv'))  # columns: ['gene', 'uniprot']
    df_lr = pd.read_csv(os.path.join(df_dir, 'lrpairs.tsv'), sep='\t')  # columns: ['ligand', 'receptor', 'species']
    df_lr = df_lr.query('species == @species')

    # columns: ['ligand', 'receptor', 'species', 'ligand_uniprot', 'receptor_uniprot']
    ligand_uniprot = df_lr['ligand'].apply(lambda x: df_lr_uniprot.query('gene == @x')['uniprot'].item())
    recep_uniprot = df_lr['receptor'].apply(lambda x: df_lr_uniprot.query('gene == @x')['uniprot'].item())
    df_lr['ligand_uniprot'] = ligand_uniprot
    df_lr['receptor_uniprot'] = recep_uniprot

    """
    ############################ load potential ligand/receptor pool ############################
    """
    # columns: ['name', 'gene', 'uniprot', 'location', 'origin', 'lr_type']
    # (1) drop ”?“ gene; (2) select F1 fragment for each gene; (3) select potential ligand and receptor
    # (4) keep one gene from duplicated genes
    # df_af = pd.read_csv(os.path.join(df_dir, 'df_af_gene_uniprot_v4_final.csv'))
    # df_potential_lr = pd.read_csv(os.path.join(df_dir, 'df_potential_lr.csv'))
    # df_af = df_af.loc[df_af['name'].apply(lambda x: '-F1-' in x) & (df_af['gene'] != '?')].reset_index(drop=True)
    # df_af = df_af.merge(df_potential_lr, on=['uniprot'])
    df_potential_lr = pd.read_csv(os.path.join(df_dir, 'df_potential_lr.csv'))
    print('len df_af:', len(df_potential_lr))

    """
    ############################ generate negative samples ############################
    """
    ls = []  # positive_gene, negative_gene, label
    for l_gene, data in df_lr.groupby('ligand'):
        ligand_data = data[['ligand', 'ligand_uniprot']].values
        ligand_data = np.concatenate([ligand_data]*(negative_fold+1))
        #
        positive = data[['receptor', 'receptor_uniprot']].values
        negative = get_negative_receptor(l_gene, positive[:, 0].tolist(), df_potential_lr, fold=negative_fold)
        receptor_data = np.concatenate((positive, negative), axis=0)
        #
        label = np.array([1]*len(positive) + [0]*len(negative)).reshape(-1, 1)
        ls.append(np.concatenate((ligand_data, receptor_data, label), axis=1))

    """
    ############################ split data ############################
    """
    dataset = pd.DataFrame(
        np.concatenate(ls, axis=0),
        columns=['ligand', 'ligand_uniprot', 'receptor', 'receptor_uniprot', 'y']
    )[['ligand', 'receptor', 'ligand_uniprot', 'receptor_uniprot', 'y']]

    dataset = split_data(dataset, val_size=val_size, test_size=test_size)
    dataset.to_csv(os.path.join(df_dir, f'df_human_dataset_nfold{negative_fold}.csv'), index=False)


