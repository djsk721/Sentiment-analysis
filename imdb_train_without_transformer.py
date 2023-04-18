import os
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
import torch


# preprocessing
def preprocess(sample):
    return {
        'text': ' '.join(re.sub(r'<[^(?:/>)]+/>', ' ', sample['text']).split()),
        'label': sample['label']
    }

# imdb 데이터셋을 사용한 영화리뷰 부정 확인
data = load_dataset('imdb')
preprocessed = data.map(preprocess)

# AutoTokenizer 또한 존재하여 자동 Tokenizer 로드 가능
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

# tokenizer = 'input_ids', 'token_type_ids', 'attention_mask'
preprocessed = preprocessed.map(
    lambda sample: tokenizer(sample['text'], truncation=True),
    remove_columns=['text'],
    batched=True
)

collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(preprocessed['train'], batch_size=16, collate_fn=collator, shuffle=True)

# model 준비
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# GPU load
model.cuda()

# optimizer 로드
# fine tuning은 MNL으로 학습된 가중치가 변하기에 분리 필요
# optimizer 분리
optimizer = torch.optim.AdamW([
    {'params': model.bert.parameters(), "lr": 3e-5},
    {"params": model.classifier.parameters(), "lr": 1e-3}
])

# 학습
model.train()
for epoch in range(3):
    print(f"Epoch: {epoch}")
    for encodings in train_loader:
        encodings = {key: value.cuda() for key, value in encodings.items()}
        # encodings = {key: value for key, value in encodings.items()}
        outputs = model(**encodings)
        
        # label이 들어간 경우 자동으로 Loss 계산
        outputs.loss.backward()
        print('\rLoss: ', outputs.loss, end='')
        optimizer.zero_grad(set_to_none=False)