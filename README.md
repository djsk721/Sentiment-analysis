# Sentiment-analysis

## 1. imdb
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
})
```

    - 데이터에 대한 긍, 부정 여부가 label에 0과 1의 형태로 담겨 있음

### - 모델
#### BERT에서 사용한 MLM을 이용한 언어모델 Pretraining 

## 2. Naver 영화 리뷰
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
})
```

    - 데이터에 대한 긍, 부정 여부가 label에 0과 1의 형태로 담겨 있음

### - 모델
#### BERT에서 사용한 MLM을 이용한 언어모델 Pretraining

## 3. AI-HUB 감성 대화 말뭉치
### - 데이터셋
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 52443
    })
    valid: Dataset({
        features: ['text', 'label'],
        num_rows: 5828
    })
})
```

    - 예시에 대한 6가지 감정(불안, 분노, 상처, 슬픔, 당황, 기쁨)이 label에 0~5의 형태로 담겨 있음

### - 모델
#### distilbert 사전학습 모델인 monologg/distilkobert 사용


# Reference
