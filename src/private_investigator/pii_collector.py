import math
import re

from tqdm import tqdm

import torch
import flair
from flair.data import Sentence
from flair.models import SequenceTagger


class PIICollector():
    """
    Collector which can collect email, phone, and names from texts.
    """

    def __init__(self, pii_type):
        if pii_type not in ['email', 'phone', 'name']:
            raise ValueError(f'Invalid PII type: {pii_type}')
        self.pii_type = pii_type
        if pii_type == 'name':
            flair.device = torch.device('cuda') 
            self.tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

    def get_email(self, texts):
        emails = []
        regex = r'(([\w\-]+\.)*[\w\-]+@([\w\-]+\.)+[\w\-]+)'
        for text in texts:
            if not isinstance(text, str):
                text = text['text']
            for match in re.findall(regex, text):
                if match[0] != '':
                    emails.append(match[0])
        return emails

    def get_phone(self, texts):
        phones = []
        regex = r'(?=(\d{3}[ .-]\d{3}[ .-]\d{4}|\(\d{3}\)[ .-]\d{3}[ .-]\d{4}))'
        for text in texts:
            if not isinstance(text, str):
                text = text['text']
            for match in re.findall(regex, text):
                phones.append(match)
        return phones

    def get_name(self, texts):
        chunk_size = 5000
        result_list = []
        for text in tqdm(texts):
            if isinstance(texts, dict):
                text = text['text']
            if len(text)< chunk_size:
                try:
                    sentences=[Sentence(text)]
                except:
                    continue
            else:
                sentences = []
                num_batch = math.ceil(len(text)/chunk_size)
                for i in range(num_batch):
                    try:
                        sentences.append(Sentence(text[i*chunk_size:(i+1)*chunk_size+20]))
                    except:
                        continue
            try:
                self.tagger.predict(sentences, mini_batch_size=1)
            except:
                continue

            for sentence in sentences:
                for entity in sentence.get_spans('ner'):
                    if any([x.to_dict()['value'] == "PERSON" for x in entity.get_labels()]):
                        result_list.append(entity.text)
            torch.cuda.empty_cache()

        return result_list 


    def get_pii(self, texts):
        if self.pii_type == 'email':
            return self.get_email(texts)
        if self.pii_type == 'phone':
            return self.get_phone(texts)
        if self.pii_type == 'name':
            return self.get_name(texts)
        else:
            raise ValueError(self.pii_type)
        
