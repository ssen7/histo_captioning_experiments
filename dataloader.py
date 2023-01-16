import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms

import random
import openslide
import h5py

from HIPT.HIPT_4K.hipt_4k import HIPT_4K
from HIPT.HIPT_4K.hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from HIPT.HIPT_4K.hipt_heatmap_utils import *

class PreLoadedReps_v2(Dataset):

    def __init__(self, df_path, dtype='train', transform=None, target_transform=None):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        self.df=df[df.dtype==dtype]
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        svs_path=self.df.iloc[idx]['svs_path']
        patch_path=self.df.iloc[idx]['patch_path']
        reps_path=self.df.iloc[idx]['reps_path']
        idx_tokens=self.df.iloc[idx]['idx_tokens']
        caplens=self.df.iloc[idx]['caplens']

        rep_tensors = torch.load(reps_path)
        caption_tensor = torch.LongTensor(idx_tokens)
        caplen = torch.LongTensor([caplens])
        
        if self.dtype=='train':
#             rep_tensor = rep_tensors[random.randint(0, len(rep_tensors)-1),:,:]
            rep_tensor=rep_tensors
            return rep_tensor, caption_tensor, caplen
        else:
            rep_tensor=rep_tensors
            all_captions=torch.LongTensor([idx_tokens])
            return rep_tensor, caption_tensor, caplen, all_captions

class PreLoadedReps(Dataset):

    def __init__(self, df_path, dtype='train', transform=None, target_transform=None):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        self.df=df[df.dtype==dtype]
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        svs_path=self.df.iloc[idx]['svs_path']
        patch_path=self.df.iloc[idx]['patch_path']
        reps_path=self.df.iloc[idx]['reps_path']
        idx_tokens=self.df.iloc[idx]['idx_tokens']
        caplens=self.df.iloc[idx]['caplens']

        rep_tensors = torch.load(reps_path)
        caption_tensor = torch.LongTensor(idx_tokens)
        caplen = torch.LongTensor([caplens])
        
        if self.dtype=='train':
#             rep_tensor = rep_tensors[random.randint(0, len(rep_tensors)-1),:,:]
            rep_tensor=torch.mean(rep_tensors,dim=0)
            return rep_tensor, caption_tensor, caplen
        else:
            rep_tensor=torch.mean(rep_tensors,dim=0)
            all_captions=torch.LongTensor([idx_tokens])
            return rep_tensor, caption_tensor, caplen, all_captions
        
class PreLoadedReps_old(Dataset):

    def __init__(self, df_path, tokenizer, transform=None, target_transform=None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        svs_path=self.df.iloc[idx]['svs_path']
        patch_path=self.df.iloc[idx]['patch_path']
        reps_path=self.df.iloc[idx]['reps_path']
        notes=self.df.iloc[idx]['notes']

        rep_tensors = torch.load(reps_path)
        rep_tensor = rep_tensors[random.randint(0, len(rep_tensors)-1),:,:]
        caption_tensor = self.create_caption_tensors([notes])

        return rep_tensor, caption_tensor
    
    def create_caption_tensors(self, text_list, max_length=80, text_transform=None):
        
        text_tensor_list = []
        attention_masks = []
        token_types = []
        for sent in text_list:
            encoded_dict = self.tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_length,           # Pad & truncate all sentences.
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True,   # remove warnings from printing
                   )
            text_tensor_list.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            token_types.append(encoded_dict['token_type_ids'])
            
        
        return torch.cat(text_tensor_list, dim=0), torch.cat(attention_masks, dim=0), torch.cat(token_types, dim=0)


class HIPTSVSImageDataset(Dataset):
    def __init__(self, df_path, transform=None, target_transform=None):
        self.df_path=df_path
        self.df=pd.read_csv(self.df_path)
        self.transform=transform
        self.target_transform=target_transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        svs_path=self.df.iloc[idx]['svs_path']
        patch_path=self.df.iloc[idx]['patch_path']

        slide = openslide.open_slide(svs_path)
        coords = h5py.File(patch_path, 'r')['coords']
        
        n_patches = len(coords)
        print(n_patches)
        random_patch_index = random.randint(0,n_patches-1)
        coord = coords[random_patch_index]
        model=self.get_model()

        return self.get_reps(model, slide, coord)

    def get_model(self):
        pretrained_weights256 = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
        pretrained_weights4k = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'

        if torch.cuda.is_available():
            device256 = torch.device('cuda:0')
            device4k = torch.device('cuda:1')
        else:
            device256 = torch.device('cpu')
            device4k = torch.device('cpu')


        ### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
        model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
        model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

        ### ViT_256 + ViT_4K loaded into HIPT_4K API
        model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
        model.eval()
        return model

    def get_reps(self, model, slide, coords, size=4096):
        x,y=coords
        region=slide.read_region((x,y),0,(size,size)).convert('RGB')
        x = eval_transforms()(region).unsqueeze(dim=0)
        out = model.forward(x)
        return list(out.cpu().numpy())