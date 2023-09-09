import os
import pandas as pd
import numpy as np
import anndata
import time
# from src import sprout as SPROUT

LRDB = './data/lrpairs.tsv'
PATHWAY = './data/pathways.tsv'
LR_novel = './data/lr_novel.csv'
LR_novel_mouse = './data/lr_novel_mouse.csv'


class AlphaTalkCCC:
    def __init__(
            self,
            args,
            input_data,
    ):
        self.args = args
        self.input = input_data
        self.src = './src/CytoTour.py'

        self.job_dir, self.cyto_tmp_dir = None, None

    def prepare_path(self):
        job_dir = os.path.join(self.args.tmp_dir, self.args.job_name)

        cyto_tmp_dir = os.path.join(job_dir, 'cyto')
        if not os.path.exists(cyto_tmp_dir):
            os.makedirs(cyto_tmp_dir)

        return job_dir, cyto_tmp_dir

    def run(self):
        #
        self.job_dir, self.cyto_tmp_dir = self.prepare_path()

        #
        if self.args.lrdb:
            species = self.args.species
            df_lrdb = pd.read_csv(LRDB, sep='\t', index_col=0)
            df_lrdb = df_lrdb.query('species == @species')[['ligand', 'receptor', 'species']].reset_index(drop=True)
            df_lrdb['type'] = 'LRDB'

            df_novel = pd.read_csv(self.input.lr_file)[['ligand', 'receptor', 'species', 'type']]
            df_lr = pd.concat([df_lrdb, df_novel]).drop_duplicates(subset=['ligand', 'receptor']).reset_index(drop=True)
            lr_file = os.path.join(self.job_dir, f'{species}_novel+lrdb.csv')
            df_lr.to_csv(lr_file, index=False)
        else:
            lr_file = self.input.lr_file
            df_lr = pd.read_csv(lr_file)

        print(time.strftime('%H:%M:%S'), f'CytoTour using lr_file at: {lr_file}, with length {len(df_lr)}')

        #
        cmd = 'python {} {} {} {} -o {} -s {} -n {} -f {} {} {}'.format(
            self.src,
            lr_file,
            self.input.pathway,
            self.input.st_file,
            self.cyto_tmp_dir,
            self.args.species,
            self.args.n_core,
            self.args.filtering,
            '--parallel' if self.args.parallel else '',
            '--distance_threshold' if self.args.dist_thd else ''
        )

        print(time.strftime('%H:%M:%S'), cmd)
        os.system(cmd)


# class SpatialRecon:
#     def __init__(
#             self,
#             args,
#             input_data,
#     ):
#         self.args = args
#         self.input = input_data
#
#         self.job_dir, self.sprout_tmp_dir = None, None
#
#     def prepare_path(self):
#         job_dir = os.path.join(self.args.tmp_dir, self.args.job_name)
#
#         sprout_tmp_dir = os.path.join(job_dir, 'sprout')
#         if not os.path.exists(sprout_tmp_dir):
#             os.makedirs(sprout_tmp_dir)
#
#         return job_dir, sprout_tmp_dir
#
#     def run(self, save_file):
#         self.job_dir, self.sprout_tmp_dir = self.prepare_path()
#         print(time.strftime('%H:%M:%S'), f'SPROUT using df_lr with length: {len(self.input.df_lr)}')
#         import warnings
#         warnings.filterwarnings("ignore")
#         # load df
#         st, sc = self.input.st, self.input.sc
#         sprout = SPROUT.SPROUT(
#             st_exp=st.to_df(),
#             st_coord=st.obsm['meta'],
#             weight=st.obsm['weight'],
#             sc_exp=sc.to_df(),
#             meta_df=sc.obsm['meta'],
#             cell_type_key='cell_type',
#             lr_df=self.input.df_lr,
#             save_path=self.sprout_tmp_dir
#         )
#
#         sprout.select_sc(num_per_spot=self.args.cell_num_per_spot, repeat_penalty=self.args.penalty)
#         sprout.spatial_recon(left_range=0, right_range=20, steps=10, dim=2, max_dist=1)
#
#         # sc_agg_meta
#         sprout.picked_index_df = sprout.picked_index_df.dropna(axis=0, how='any')
#         agg_meta = sprout.picked_index_df[['adj_UMAP1', 'adj_UMAP2', 'celltype']].reset_index(drop=True)
#         agg_meta.columns = ['x', 'y', 'cell_type']
#         agg_meta.index = "C" + agg_meta.index.astype('str')
#
#         # sc_agg_exp
#         agg_exp = sc[sprout.picked_index_df['sc_id']].to_df().reset_index(drop=True)
#         agg_exp.index = "C" + agg_exp.index.astype('str')
#
#         #
#         sc_st = anndata.AnnData(agg_exp)
#         sc_st.obsm['meta'] = agg_meta
#         sc_st.write(save_file, compression="gzip")
#         print(time.strftime('%H:%M:%S'), f'Export sc_st at {save_file}')
#
#         return sc_st
