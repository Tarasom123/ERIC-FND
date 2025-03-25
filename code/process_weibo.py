# 使用jieba和wikipediaapi提取微博数据集的实体和实体描述
import re
import json
import wikipediaapi
import jieba.analyse
from tqdm import tqdm
from opencc import OpenCC


def read_data(file_path):
    with open(file_path, 'r') as f:
        train_data = json.load(f)
    return train_data

def traditional_to_simplified(traditional_text):
    cc = OpenCC('t2s')
    simplified_text = cc.convert(traditional_text)
    return simplified_text

def get_entity_list(sentence):
    entity_list = jieba.analyse.textrank(sentence, topK=10, allowPOS=('ns', 'n', 'nr'))
    return entity_list

def get_wiki_page_title(entity):
    wiki_wiki = wikipediaapi.Wikipedia('zh')
    page_py = wiki_wiki.page(entity)
    return page_py.title

def get_entity_desc(entity):
    wiki_wiki = wikipediaapi.Wikipedia('zh')
    page_py = wiki_wiki.page(entity)
    desc = traditional_to_simplified(page_py.summary)
    return desc

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
def process_data(train_data):
    final_train = []
    for dic in tqdm(train_data):
        try:
            text = text_preprocessing(dic['text'],'weibo_dataset')
            entity_list = get_entity_list(text)
            wiki_list = []
            for entity in entity_list:
                if re.search(r'链接|网页|中国|天津|网友|南京|北京|记者|美国|请告|国家|纽约', entity):
                    continue
                page_title = get_wiki_page_title(entity)
                wiki_list.append(page_title)
            wiki_list = list(set(wiki_list))
            entity_desc_list = []
            ent_desc_tup = []
            for name in wiki_list:
                desc = get_entity_desc(name)
                if len(desc) > 0:
                    entity_desc_list.append(desc)
                    ent_desc_tup.append((name, desc))
        except:
            print("提取实体描述错误")
            continue
        Dic = dict()
        Dic['id'] = dic['id']
        Dic['text'] = dic['text']
        Dic['label'] = dic['label']
        Dic['desc_list'] = entity_desc_list
        Dic['ent_desc_tup'] = ent_desc_tup
        final_train.append(Dic)
    return final_train

train_data = read_data('data/weibo_dataset/train_set.json')
test_data = read_data('data/weibo_dataset/test_set.json')

final_train = process_data(train_data)
final_test = process_data(test_data)

with open("./data/weibo_dataset/train_set.json", "r") as f:
    json.dump(final_train,f)

with open("./data/weibo_dataset/test_set.json", "r") as f:
    json.dump(final_test,f)