import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import text_preprocessing,chinese_sentence_tokenize,english_sentence_tokenize

class MyDataset(Dataset):
    def __init__(
            self,
            data,
            tokenizer,
            image_transform,
            label_mapping,
            max_sent_len,
            max_sent_num,
            max_desc_sent_num,
            max_single_desc,
            data_name
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.label_mapping = label_mapping
        self._idx2label = {idx: label for label, idx in self.label_mapping.items()}
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.max_desc_sent_num = max_desc_sent_num
        self.max_single_desc = max_single_desc
        self.data_name = data_name
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        instance = self.data[index]
        return instance

    def num_classes(self) -> int:
        return len(self.label_mapping)


    def encode_batch(self, batch, max_sent_len, max_sent_num):
        padded_list = []
        mask_list = []
        for batch_tokens in batch:
            bert_tokens = [self.tokenizer(tokens, max_length=max_sent_len, truncation=True, padding='max_length',add_special_tokens=True) for tokens in batch_tokens]
            input_ids,attention_mask = [b.get('input_ids')for b in bert_tokens],[b.get('attention_mask')for b in bert_tokens] 
            padded_list.append(input_ids)
            mask_list.append(attention_mask)
        pad_list = [0] * max_sent_len
        padded_list = self.pad_to_num(padded_list, max_sent_num, pad_list)
        mask_list = self.pad_to_num(mask_list, max_sent_num, pad_list)
        return padded_list,mask_list

    def pad_to_len(self,seqs, to_len, padding):
        # 填充至最大长度
        paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
        return paddeds

    def pad_to_num(self,seqs, to_len, padding):
        # 填充至最大数量
        paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
        return paddeds


    def collate_fn(self, samples):
        padded_sent_list = []
        padded_desc_list = []
        images_list = []
        label_list = []
        for instance in samples:
            label_list.append(self.label2idx(instance['label']))
            if self.data_name in ['twitter','caobiwei_data']:
                sent_list = []
                sent = text_preprocessing(instance['text'],'twitter')
                sent_list.append(sent)
           
                desc_list = []
                for desc in instance['desc_list']:
                    count = 0
                    for sent in english_sentence_tokenize(desc):
                        sent = text_preprocessing(sent,'twitter')
                        desc_list.append(sent)
                        count += 1
                        if count == self.max_single_desc:
                            break
                    if count == self.max_single_desc:
                        break
            else:
                sent_list = []
                sent = text_preprocessing(instance['text'],'weibo')
                sent_list.append(sent)
          
                desc_list = []
                for desc in instance['desc_list']:
                    count = 0
                    for sent in chinese_sentence_tokenize(desc):
                        sent = text_preprocessing(sent,'weibo')
                        desc_list.append(sent)
                        count += 1
                        if count == self.max_single_desc:
                            break
                    if count == self.max_single_desc:
                        break
      
            img_id = instance['id']
         
            if 'jpg' or 'jpf' in img_id:
                img_name = f'../data/{self.data_name}/Images/{img_id}'
            else:
                img_name = f'../data/{self.data_name}/Images/{img_id}.jpg'
            if self.data_name in ['twitter','caobiwei_data']:
                tmp = img_id.split('.')[0]
                img_name = f'../data/{self.data_name}/Images/{tmp}.jpg'
            image = Image.open(img_name).convert("RGB")
            image = self.image_transform(image)

            
            images_list.append(image)
            padded_sent_list.append(sent_list)
            padded_desc_list.append(desc_list)

        batch = {}
        label_list = torch.tensor(label_list)
        batch["label_list"] = label_list


        padding_sent,text_mask = self.encode_batch(batch=padded_sent_list, max_sent_len=self.max_sent_len,
                                            max_sent_num=self.max_sent_num)
  
        padding_sent = sum(padding_sent, [])
        padding_sent = torch.tensor(padding_sent)
        batch["texts"] = padding_sent

        text_mask = sum(text_mask, [])
        text_mask = torch.tensor(text_mask)
        batch["text_mask"] = text_mask

        padding_desc,desc_mask = self.encode_batch(batch=padded_desc_list, max_sent_len=30,
                                          max_sent_num=self.max_desc_sent_num)
        padding_desc = sum(padding_desc, [])
        padding_desc = torch.tensor(padding_desc)
        batch["descs"] = padding_desc

        desc_mask = sum(desc_mask, [])
        desc_mask = torch.tensor(desc_mask)
        batch["desc_mask"] = desc_mask

        batch['images'] = images_list

        batch['images'] = torch.stack(images_list)


        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]