import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer

class BertEncoder(nn.Module):
    
    def __init__(self, bert, output_size=300, dropout_p=0.5):
        super(BertEncoder, self).__init__()
        self.bert = bert
        self.text_enc_fc1 = torch.nn.Linear(768, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fine_tune()
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc1(out['pooler_output']))
        )    
        return x
    
    def fine_tune(self):
        """
        固定参数
        """
        for p in self.bert.parameters():
            p.requires_grad = False


class ResNetEncoder(nn.Module):
    def __init__(self, img_fc1_out=2742, img_fc2_out=512, dropout_p=0.4, fine_tune_module=True):
        super(ResNetEncoder, self).__init__()
        self.fine_tune_module = fine_tune_module

        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(torch.load('./pretrain_vector/resnet50-19c8e357.pth'))

        modules = list(resnet.children())[:-1]
        self.vis_encoder = nn.Sequential(*modules)

        self.vis_enc_fc1 = torch.nn.Linear(2048, img_fc1_out) 
        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)
        self.dropout = nn.Dropout(dropout_p)
        self.fine_tune()
        
    def forward(self, images):
        """
        :参数: images, tensor (batch_size, 3, image_size, image_size)
        :返回: encoded images
        """
        x = self.vis_encoder(images)
        x = torch.flatten(x, 1) 
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )
        return x
    
    def fine_tune(self):
        """
        允许或阻止resnet的梯度计算。
        """
        for p in self.vis_encoder.parameters():
            p.requires_grad = False

 
        if self.fine_tune_module:
            for p in list(self.vis_encoder.children())[-1].parameters():
                p.requires_grad = True
                
class VggEncoder(nn.Module):

    def __init__(self, img_fc1_out=2742, img_fc2_out=512, dropout_p=0.4, fine_tune_module=True):
        super(VggEncoder, self).__init__()
        self.fine_tune_module = fine_tune_module
  
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load('./pretrain_vector/vgg19-dcbb9e9d.pth'))
 
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        self.vis_encoder = vgg
        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)
        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)
        self.dropout = nn.Dropout(dropout_p)
        self.fine_tune()

    def forward(self, images):
        x = self.vis_encoder(images)
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )
        return x

    def fine_tune(self):

        for p in self.vis_encoder.parameters():
            p.requires_grad = False

        for c in list(self.vis_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module