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
![cnn](figs/qc_cnn_train.png)

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

#### GRU-CNN training
![gru-cnn](figs/qc_gru_cnn_train.png)

    Epoch 3:
        train cost: 0.000631529912567
        train acc: 0.993626882966
        dev acc: 0.984


#### RCNN training
![rcnn](figs/qc_rcnn_train.png)

    Epoch 1:
        train cost: 0.00179642121795
        train acc: 0.980011587486
        dev acc: 0.98
    Epoch 2:
        train cost: 0.000787947041509
        train acc: 0.992757821553
        dev acc: 0.982
    Epoch 3:
        train cost: 0.000631529912567
        train acc: 0.993626882966
        dev acc: 0.984



## NLPCC 2017 News headline Classification
https://github.com/FudanNLP/nlpcc2017_news_headline_categorization

| type | objective | average length | class | train/dev/test | Lan
| ------ | ------ | ------ | ------ | ------ |  ------ |
| sentence | news headline | 10 | 6 | 0/0/0 |Chinese




