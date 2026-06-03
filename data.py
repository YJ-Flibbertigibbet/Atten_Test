from datasets import load_dataset
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

# Load the SST2 dataset：英文电影评论二分类
'''
{'sentence': 'hide new secretions from the parental units ', 'label': 0, 'idx': 0}
'''

sst2_dataset = load_dataset("glue", "sst2")
sst2_textCol="sentence"
sst2_labelCol="label"


# Load the Yelp Polarity dataset：英文电影评论二分类
'''
{'text': "Unfortunately, the frustration of being Dr. Goldberg's patient...", 'label': 0}
'''

yelp_dataset = load_dataset("yelp_polarity")
yelp_textCol="text"
yelp_labelCol="label"


# Load the ChnSentiCorp dataset：中文电影评论二分类
'''
file= /root/.cache/modelscope/hub/datasets/AiNiklaus/ChnSentiCorp
'''





















# ceshi

# print(sst2_train_dataset[0], yelp_train_dataset[0])
'''
sst2:{'sentence': 'hide new secretions from the parental units ', 'label': 0, 'idx': 0}
yelp:{'text': "Unfortunately, the frustration of being Dr. Goldberg's patient...", 'label': 0}
'''

