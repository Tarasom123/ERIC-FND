import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from network import Network
from torch.nn.functional import softplus
from feature_extraction import BertEncoder,ResNetEncoder,VggEncoder
from utils import split_s
from coattention import CoAttention
import time
import numpy as np
class EncodingPart(nn.Module):

    def __init__(
            self,
            embeddings,
            data_name,
            shared_image_dim=128,
            shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.text_enc = BertEncoder(embeddings)
        if data_name == 'twitter':
            self.vision_enc = VggEncoder()
        else:
            self.vision_enc = ResNetEncoder()
        self.shared_text_linear = nn.Sequential(
            nn.Linear(300, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text, mask, image = None):
        if image is None:
            text_encoding = self.text_enc(text,mask)
            text_shared = self.shared_text_linear(text_encoding)
            return text_shared
        text_encoding = self.text_enc(text,mask)
        image_encoding = self.vision_enc(image)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image_encoding)
        return text_shared, image_shared


class UnimodalDetection(nn.Module):

    def __init__(self, shared_dim=128, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, text_encoding, image_encoding=None):
        if image_encoding is None:
            text_prime = self.text_uni(text_encoding)
            return text_prime
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class CrossModule4Batch(nn.Module):

    def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    
    def forward(self, text, image):
      
        corre_dim = text.shape[1]
        similarity_t2v = torch.matmul(text, image.T) / math.sqrt(corre_dim) 
        similarity_v2t = torch.transpose(similarity_t2v,0,1) 
        correlation_t2v = self.softmax(similarity_t2v) # torch.Size([batch, batch])
        correlation_v2t = self.softmax(similarity_v2t)
     
        text = torch.matmul(correlation_t2v,text)# [batch,64]
        image = torch.matmul(correlation_v2t,image)# [batch,64]
        correlation_out = torch.matmul(text.unsqueeze(2), image.unsqueeze(1))
        correlation_out = self.pooling(correlation_out).squeeze()
        correlation_out = self.c_specific_2(correlation_out)
 
        return correlation_out

def save_features(features, filename):
    np.save(filename, features.detach().cpu().numpy())
class DetectionModule(nn.Module):
    def __init__(self,embeddings,data_name,feature_dim=64 + 128 + 128, h_dim=64):
        super(DetectionModule, self).__init__()
        self.encoding = EncodingPart(embeddings,data_name)
        self.uni_se = UnimodalDetection(prime_dim=64)
        self.uni_repre = UnimodalDetection(prime_dim=128)
        self.senet = Network(64, 128, 19, 3)
        self.cross_module = CrossModule4Batch()
        self.co_attention = CoAttention(64, 64)
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, 2)
        )

     
    def forward(self, text_raw, text_mask, image_raw, text, image, desc, desc_mask):  
    
        text_prime,image_prime = self.encoding(text_raw, text_mask, image_raw)


        wiki_starttime = time.process_time()
      

        desc_prime = self.encoding(desc, desc_mask)
    

        batch_size = text_prime.shape[0]
        desc_prime = split_s(desc_prime,batch_size)
        desc_prime = torch.stack(desc_prime) 
    

    
        text_prime = self.co_attention(text_prime.unsqueeze(1),desc_prime)

        
        wiki_endtime = time.process_time()
        print("wiki time:", wiki_endtime - wiki_starttime)
      
        text_se, image_se = self.uni_se(text_prime, image_prime)

        text_prime, image_prime = self.uni_repre(text_prime, image_prime)

 
        correlation = self.cross_module(text, image)
 
        text_se, image_se, corre_se = text_se.unsqueeze(-1), image_se.unsqueeze(-1), correlation.unsqueeze(-1)
        attention_score = self.senet(torch.cat([text_se, image_se, corre_se], -1)) 
   
        text_final  = text_prime * attention_score[:, 0].unsqueeze(1)
        img_final   = image_prime * attention_score[:, 1].unsqueeze(1)
        corre_final = correlation * attention_score[:, 2].unsqueeze(1)
        
        final_corre = torch.cat([text_final, img_final, corre_final], 1)


        pre_label = self.classifier_corre(final_corre)

        save_features(final_corre, "final_corre.npy")



        return pre_label, final_corre