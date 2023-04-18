# Sentiment-analysis

## imdb
### - 데이터셋
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
```

    - 데이터에 대한 긍, 부정 여부가 label에 0과 1의 형태로 담겨 있음

### - 모델
#### BERT에서 사용한 MLM을 이용한 언어모델 Pretraining 

![image.png](attachment:image.png)

## Naver 영화 리뷰
### - 데이터셋
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 150000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
```

    - 데이터에 대한 긍, 부정 여부가 label에 0과 1의 형태로 담겨 있음

### - 모델
#### BERT에서 사용한 MLM을 이용한 언어모델 Pretraining