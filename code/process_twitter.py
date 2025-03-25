# 使用tagme和wikipediaapi提取twitter数据集的实体和实体描述
import re
import json
import tagme
import socket
import urllib3
import requests
import pandas as pd
import wikipediaapi
from tqdm import tqdm

tagme.GCUBE_TOKEN ="14862c95-3864-4609-a7b4-3855137e388f-843339462" # 需要注册tagme账号获取token
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent=user_agent
)

def read_data(file_path):
    with open(file_path, 'r') as f:
        train_data = json.load(f)
    return train_data

def text_preprocessing(text,data_name):
    """
    数据清洗
    """
    if data_name == 'twitter':
        # 去除 '@name'
        text = re.sub(r'(@.*?)[\s]', ' ', text)
        #  替换'&amp;'成'&'
        text = re.sub(r'&amp;', '&', text)
        # 删除尾随空格
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        text = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", text)
    return text

# 获取实体和实体描述
def process_text(text):
    entity_desc_list = []
    try:
        annotations = tagme.annotate(text)
        if annotations is None:
            return None
        entitylist = [str(ann).split(" -> ")[1].split(" (score: ")[0] for ann in annotations.get_annotations(0.3)]
        for entity in set(entitylist):
            page_py = wiki_wiki.page(entity)
            try:
                entity_desc_list.append(page_py.summary)
            except (json.decoder.JSONDecodeError, TimeoutError, socket.timeout, urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout):
                print("small, Error occurred")
                continue
        return entity_desc_list
    except:
        print("提取实体描述错误")
        return None


def process_data(train_data):
    final_train = []
    for i in tqdm(range(1, 135)):
        batch_train = []
        for tr in train_data[i * 100: (i + 1) * 100]:
            # 清洗数据
            text = text_preprocessing(tr['text'], 'twitter')
            # 获取实体描述
            entity_desc_list = process_text(text)
            if entity_desc_list is not None:
                tr['desc_list'] = entity_desc_list
                batch_train.append(tr)
        final_train.extend(batch_train)
    return final_train

train_data = read_data('data/twitter/train_set.json')
test_data = read_data('data/twitter/test_set.json')

final_train = process_data(train_data)
final_test = process_data(test_data)

with open("./data/twitter/train_set.json", "r") as f:
    json.dump(final_train,f)

with open("./data/twitter/test_set.json", "r") as f:
    json.dump(final_test,f)