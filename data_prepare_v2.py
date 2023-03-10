import pandas as pd
import numpy as np
import os

def create_df_tokens(df_path):

    df =  pd.read_csv(df_path)


def main():
    svs_path='/project/GutIntelligenceLab/ss4yd/gtex_data/actual_useful/'
    paths = {x.split('/')[-1].split('.')[0]:os.path.join(svs_path,x) for x in os.listdir(svs_path)}
    patch_path='/project/GutIntelligenceLab/ss4yd/gtex_data/patches/patches/'
    patch_paths = {x.split('/')[-1].split('.')[0]:os.path.join(patch_path,x) for x in os.listdir(patch_path)}
    reps_path='/project/GutIntelligenceLab/ss4yd/gtex_data/hipt4k_256cls_reps/'
    reps_paths = {x.split('/')[-1].split('.')[0]:os.path.join(reps_path,x) for x in os.listdir(reps_path)}

    df=pd.read_csv('/home/ss4yd/vision_transformer/captioning_vision_transformer/GTExPortal.csv')
    df=df[['Tissue Sample ID','Tissue','Pathology Notes']]
    df.columns=['pid', 'tissue_type','notes']

    df['svs_path']=df['pid'].map(paths)
    df['patch_path']=df['pid'].map(patch_paths)
    df['reps_path']=df['pid'].map(reps_paths)
    df=df.dropna()

    print(df.head())
    print(df.shape)

    df.to_csv('./prepared_prelim_data_cls256.csv',index=False)

if __name__ == "__main__":
    main()