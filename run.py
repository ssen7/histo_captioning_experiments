from dataloader import SVSImageDataset

df_path='/home/ss4yd/vision_transformer/captioning_vision_transformer/prepared_prelim_data.csv'
dataset = SVSImageDataset(df_path=df_path)
print(len(dataset))
print(dataset[0])