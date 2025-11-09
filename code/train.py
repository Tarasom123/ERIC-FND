import os
import sys
import math
import json
import torch
import random
import torchvision
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer,BertModel
from dataset import MyDataset
from model import DetectionModule
from clip import CLIP
import torch.fft
import torch.nn as nn
import time


# Configs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_EPOCH = 20
seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import pandas as pd
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def save_results_to_csv(tokenizer, texts, preds, labels, csv_path="test_results.csv"):

    texts_str = [tokenizer.decode(text, skip_special_tokens=True) for text in texts]


    data = {
        "Text": texts_str,
        "Predicted Label": preds,
        "True Label": labels
    }
    df = pd.DataFrame(data)

 
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
def save_features(features, filename):
    features = torch.cat(features, dim=0)
    np.save(filename, features.detach().cpu().numpy())
def train(data_name,epoch,LR):
    # ---  Load Config  ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    batch_size = BATCH_SIZE
    num_epoch = int(epoch)

    if data_name == 'weibo':
        encoding = 'utf-8'
    else:
        encoding = None
    # ---  Load Data  ---
    with open(f'../data/{data_name}/train_set.json', 'r', encoding=encoding) as f:
        train_data = json.load(f)
    with open(f'../data/{data_name}/test_set.json', 'r', encoding=encoding) as f:
        test_data = json.load(f)
        
    print(f'{data_name}数据集加载完毕...')
    print('训练集大小：',len(train_data))
    print('测试集大小：',len(test_data))
    label_mapping = {
        'real': 0,
        'fake': 1}
    
    if data_name == 'twitter':
        tokenizer = BertTokenizer.from_pretrained('pretrain_vector/bert-base-uncased', do_lower_case=True)
        bert = BertModel.from_pretrained('pretrain_vector/bert-base-uncased')
        max_sent_len = 30
        max_desc_sent_num = 6
        max_single_desc = 2
        
    else:
        tokenizer = BertTokenizer.from_pretrained('pretrain_vector/bert-base-chinese')
        bert = BertModel.from_pretrained('pretrain_vector/bert-base-chinese')
        max_sent_len = 100
        max_desc_sent_num = 10
        max_single_desc = 2


    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    tr_dataset = MyDataset(train_data, tokenizer, image_transform, label_mapping, max_sent_len=max_sent_len, max_sent_num=1, 
                           max_desc_sent_num=max_desc_sent_num, max_single_desc=max_single_desc,data_name=data_name)
    te_dataset = MyDataset(test_data, tokenizer, image_transform, label_mapping, max_sent_len=max_sent_len,max_sent_num=1, 
                           max_desc_sent_num=max_desc_sent_num, max_single_desc=max_single_desc,data_name=data_name)
    
    train_loader = DataLoader(tr_dataset, batch_size=batch_size, collate_fn=tr_dataset.collate_fn, shuffle=True)
    test_loader = DataLoader(te_dataset, batch_size=batch_size, collate_fn=te_dataset.collate_fn, shuffle=False)
    
   
    vis_enc = 'resnet'
    train_log_filename = f"../logs/{data_name}/train4{data_name}_log_1208.txt"
    with open(train_log_filename, "w") as f:
        f.write(f'lr={LR}, max_sent_len={max_sent_len}, max_desc_sent_num={max_desc_sent_num}, max_single_desc={max_single_desc}, vis_enc={vis_enc} \n')


    detection_module = DetectionModule(bert,data_name)
    detection_module.to(device)

    clip_module = CLIP(64,bert,data_name)  # img_out,text_out
    clip_module.to(device)

    loss_func_clip = torch.nn.CrossEntropyLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
   
    optimizer_task_clip = torch.optim.AdamW(
        clip_module.parameters(), lr=LR, weight_decay=5e-4)
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=LR, weight_decay=1e-5) 

    total_steps = (len(train_data)//batch_size+1)*num_epoch
    scheduler_task_clip = CosineAnnealingLR(optimizer_task_clip, T_max=total_steps)
    scheduler_task_detection = CosineAnnealingLR(optim_task_detection, T_max=total_steps)

    # ---  Model Training  ---
    loss_detection_total = 0
    best_acc = 0
    step = 0

    best_all_texts = []
    best_all_images = []
    best_all_preds = []
    best_all_labels = []
    best_all_final_corre = []

    for epoch in range(num_epoch):
        detection_module.train()
        clip_module.train()
        corrects_pre_detection = 0
        loss_clip_total = 0
        loss_detection_total = 0
        detection_count = 0

        for i, batch in tqdm(enumerate(train_loader)):
            text = batch["texts"].to(device)
            mask = batch["text_mask"].to(device)
            desc = batch["descs"].to(device)
            desc_mask = batch["desc_mask"].to(device)
            image = batch['images'].to(device)
            label = batch['label_list'].to(device)
            image_aligned, text_aligned = clip_module(text, mask, image)
            logits = torch.softmax(torch.matmul(image_aligned, text_aligned.T),dim=1)
            labels = torch.arange(image.size(0))
            labels = labels.to(device)

            
            optimizer_task_clip.zero_grad()
            loss_clip_i = loss_func_clip(logits, labels)
            loss_clip_t = loss_func_clip(logits.T, labels)
            loss_clip = (loss_clip_i + loss_clip_t) / 2
            loss_clip.backward()
            step += 1
            optimizer_task_clip.step()
            

            image_aligned, text_aligned = clip_module(text,mask,image)
            pre_detection, _ = detection_module(text, mask, image, text_aligned, image_aligned, desc, desc_mask)
            loss_detection = loss_func_detection(pre_detection, label)
            
            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()
            
            scheduler_task_clip.step()
            scheduler_task_detection.step()

            pre_label_detection = pre_detection.argmax(1)
            corrects_pre_detection += pre_label_detection.eq(label.view_as(pre_label_detection)).sum().item()

            # ---  Record  ---
            loss_detection_total += loss_detection.item() * image.shape[0]
            detection_count += image.shape[0]

        loss_detection_train = loss_detection_total / detection_count
        acc_detection_train = corrects_pre_detection / detection_count
        timer_train = time.process_time()
        # ---  Test  ---

        acc_detection_test, loss_detection_test, cm_detection, cr_detection,all_texts, all_images, all_preds, all_labels, all_final_corre = test(clip_module, detection_module,
                                                                                   test_loader)
        timer_test = time.process_time()
        # ---  Output  ---
        print("total time:", timer_test - timer_start)
        print("train time:", timer_start - timer_train)
        print("test time:", timer_train- timer_test)
        print('---  TASK1 CLIP  ---')
        print('[Epoch: {}], losses: {}'.format(epoch+1, loss_clip_total / step))
        
        print('---  TASK2 Detection  ---')
        if acc_detection_test > best_acc:
            best_acc = acc_detection_test
            print(cr_detection)
            torch.save(clip_module.state_dict(), f"../saved_models/{data_name}/best_clip_module_E1208.pth")
            torch.save(detection_module.state_dict(), f"../saved_models/{data_name}/best_detection_model_E1208.pth")
            best_all_texts = all_texts
            best_all_labels = all_labels
            best_all_preds = all_preds
            best_all_final_corre = all_final_corre
        print(
            "EPOCH = %d \n acc_detection_test = %.3f \n  best_acc = %.3f \n loss_detection_train = %.3f \n loss_detection_test = %.3f \n" %
            (epoch + 1, acc_detection_test, best_acc, loss_detection_train, loss_detection_test)
        )

        print('---  TASK2 Detection Confusion Matrix  ---')
        print('{}\n'.format(cm_detection))
        
        # \n acc_detection_train = %.3f 
        with open(train_log_filename, "a") as f:
            f.write(
            "EPOCH = %d \n acc_detection_test = %.3f \n  best_acc = %.3f \n loss_detection_train = %.3f \n loss_detection_test = %.3f \n Classification Report:\n%s\n" %
            (epoch + 1, acc_detection_test, best_acc, loss_detection_train, loss_detection_test, cr_detection)
        )
    save_results_to_csv(tokenizer, best_all_texts, best_all_preds, best_all_labels)
    save_features(best_all_final_corre, "all_final_corre.npy")
        

def test(clip_module, detection_module, test_loader):
    clip_module.eval()
    detection_module.eval()

    device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_clip = torch.nn.CrossEntropyLoss()
  
    detection_count = 0
    loss_clip_total = 0
    loss_detection_total = 0
    detection_label_all = []
    detection_pre_label_all = []




    all_texts = []
    all_images = []
    all_preds = []
    all_labels = []
    all_final_corre = []
    


    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            text = batch["texts"].to(device)
            mask = batch["text_mask"].to(device)
            desc = batch["descs"].to(device)
            desc_mask = batch["desc_mask"].to(device)
            image = batch['images'].to(device)
            label = batch['label_list'].to(device)

            image_aligned, text_aligned =  clip_module(text, mask, image)

            logits = torch.softmax(torch.matmul(image_aligned, text_aligned.T),dim=1)
            labels = torch.arange(image.size(0))
            labels = labels.to(device)
            loss_clip = loss_func_clip(logits, labels)
            pre_detection, final_coore = detection_module(text, mask, image, text_aligned, image_aligned, desc, desc_mask)
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)

            # ---  Record  ---

            loss_clip_total += loss_clip.item()
            loss_detection_total += loss_detection.item() * image.shape[0]
            detection_count += image.shape[0]

            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

       
            all_texts.extend(text.detach().cpu().numpy())
            all_images.extend(image.detach().cpu().numpy())
            all_preds.extend(pre_label_detection.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())
            all_final_corre.append(final_coore)

        loss_detection_test = loss_detection_total / detection_count

        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)
        cr_detection = classification_report(detection_pre_label_all, detection_label_all,
                                             target_names=['Real News', 'Fake News'], digits=3)

    return acc_detection_test, loss_detection_test, cm_detection, cr_detection, all_texts, all_images, all_preds, all_labels,all_final_corre

if __name__ == '__main__':
    timer_start = time.process_time()
    data = 'weibo'
    epoch = 8
    lr = 0.001
    train(data,epoch,lr)