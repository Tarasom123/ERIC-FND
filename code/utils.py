import re
import jieba
from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        token = token.lower()
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
            self, batch: List[List[List[str]]], max_sent_len: int = None, max_sent_num: int = None
    ) -> List[List[List[int]]]:

        padded_list = []
        for batch_tokens in batch:
            batch_ids = [self.encode(tokens) for tokens in batch_tokens]
            to_len = max(len(ids) for ids in batch_ids) if max_sent_len is None else max_sent_len
            padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
            padded_list.append(padded_ids)

        pad_list = [self.pad_id] * max_sent_len
        padded_list = pad_to_num(padded_list, max_sent_num, pad_list)
        return padded_list


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:

    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds


def pad_to_num(seqs: List[List[List[int]]], to_len: int, padding: List) -> List[List[List[int]]]:

    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds


def text_preprocessing(text,data_name):

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


def chinese_sentence_tokenize(text):
    '''中文分句'''
    sentences = []
    seg_list = jieba.cut(text, cut_all=False)
    current_sentence = []

    for word in seg_list:
        current_sentence.append(word)
        if word in ['。', '！', '？']:
            sentences.append(''.join(current_sentence))
            current_sentence = []

    if current_sentence:
        sentences.append(''.join(current_sentence))

    return sentences


def english_sentence_tokenize(text):

    sentence_endings_regex = re.compile(r"""

        (?<!\w\.\w.)    
        (?<=[.!?])     
        \s              
        (?![a-zA-Z0-9])  
    """, re.VERBOSE)
    

    sentences = sentence_endings_regex.split(text)
 
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def split_s(a, n):  
    k, m = divmod(len(a), n)  
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))