# Method
## 1-EMNLP2014 Convolutional Neural Networks for Sentence Classification

## 2-AAAI2015 Recurrent Convolutional Neural Networks for Text Classification

## 3-NAACL2016 Hierarchical Attention Networks for Document Classification

## 4-ICLR2017 A Structured Self-attentive Sentence Embedding

# Experiments

## Question Classification (QC 2002)
http://cogcomp.org/Data/QA/QC/

| type | objective | average length | class | train/test | Lan|
| ------ | ------ | ------ | ------ | ------ | ------ |
| sentence | question types | 10 | 6 | 15452/500 | English|

#### kim cnn training
![kim-cnn](figs/qc_cnn_train.png)

    Epoch 2:
        cost: 0.00132063299802
        train acc: 0.989860950174
        dev acc: 0.974

#### GRU training
![gru](figs/qc_gru_train.png)

    Epoch 1:
        cost: 0.000869356350728
        train acc: 0.992178447277
        dev acc: 0.984



## NLPCC 2017 News headline Classification
https://github.com/FudanNLP/nlpcc2017_news_headline_categorization

| type | objective | average length | class | train/dev/test | Lan
| ------ | ------ | ------ | ------ | ------ |  ------ |
| sentence | news headline | 10 | 6 | 0/0/0 |Chinese




