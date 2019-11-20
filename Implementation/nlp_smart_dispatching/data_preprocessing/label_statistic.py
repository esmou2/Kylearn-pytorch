import pandas as pd
pd.options.display.max_rows = 999

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['font.size'] = 15

data = pd.read_csv('/home/oem/Projects/Kylearn-pytorch/Implementation/nlp_smart_dispatching/data/8910_split_loc_dpt.csv', encoding='gb18030')
location_ltp = data['ltp_loc'].value_counts()
location_jieba = data['jieba_loc'].value_counts()
department_ltp = data['ltp_dpt'].value_counts()
department_jieba = data['jieba_dpt'].value_counts()

# plot = location_ltp.plot(kind='bar')
# plt.show()
# plot = location_jieba.plot(kind='bar')
# plt.show()
# plot = department_ltp.plot(kind='bar')
# plt.show()
# plot = department_jieba.plot(kind='bar')
# plt.show()

location_ltp_percent = data['ltp_loc'].value_counts() / data['ltp_loc'].value_counts().sum()
department_ltp_percent = data['ltp_dpt'].value_counts() / data['ltp_dpt'].value_counts().sum()

plot = location_ltp_percent.plot(kind='bar')
plt.show()
plot = department_ltp_percent.plot(kind='bar')
plt.show()