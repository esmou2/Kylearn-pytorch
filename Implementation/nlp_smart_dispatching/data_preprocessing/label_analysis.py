import pandas as pd
import requests
from bs4 import BeautifulSoup
import jieba.posseg as pseg
import numpy as np
pd.options.display.max_rows = 999

def ltp_split(input):
    invalid_list = ['其他单位', '省外单位', 'nan', '省级单位', '除海口外的市县', '无效归属', '无效数据', '政府单位']

    if input in invalid_list:
        return input

    payload = {}
    payload['s'] = input
    payload['f'] = 'xml'
    payload['t'] = 'pos'
    response = requests.post("http://127.0.0.1:12345/ltp", data=payload)
    # docker run -d -p 12345:12345 ce0140dae4c0 ./ltp_server --last-stage all
    # sudo docker run -d -p 12345:12345 b42cadd12873 ./ltp_server --last-stage all
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup)
    word_tags = soup.findAll('word')

    location = []
    dept = []

    for word in word_tags:
        if word['pos'] == 'wp':
            continue
        elif word['pos'] == 'ns':
            location.append(word['cont'])
        else:
            dept.append(word['cont'])

    location = ','.join(location)
    dept = ','.join(dept)

    return location, dept

def jieba_split(input):

    invalid_list = ['其他单位', '省外单位', 'nan', '省级单位', '除海口外的市县', '无效归属', '无效数据', '政府单位', np.nan]

    if input in invalid_list:
        return str(input)

    location = []
    dept = []

    words = pseg.cut(input)

    for word, flag in words:
        if flag == 'x':
            continue
        elif flag in ['nz', 'ns']:
            location.append(word)
        else:
            dept.append(word)


    location = ','.join(location)
    dept = ','.join(dept)

    return location, dept

file_path = '../data/2019'

data_1 = pd.read_csv(file_path + '08.csv', encoding='gb18030')
data_2 = pd.read_csv(file_path + '09.csv', encoding='gb18030')
data_3 = pd.read_csv(file_path + '10.csv', encoding='gb18030')

data = new = pd.concat([data_1, data_2, data_3], axis = 0, ignore_index=True)

labels = data['第一级名称.1']

ltp = pd.DataFrame(columns=['ltp_loc', 'ltp_dpt'])
ltp['ltp_loc'], ltp['ltp_dpt'] = zip(*labels.apply(ltp_split))

jieba = pd.DataFrame(columns=['jieba_loc', 'jieba_dpt'])
jieba['jieba_loc'], jieba['jieba_dpt'] = zip(*labels.apply(jieba_split))

new = pd.concat([data, ltp, jieba], axis = 1)

new.to_csv('../data/8910_split_loc_dpt.csv', encoding = 'gb18030')