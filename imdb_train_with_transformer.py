import os
import re
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


# preprocessing
def preprocess(sample):
    return {
        'text': ' '.join(re.sub(r'<[^(?:/>)]+/>', ' ', sample['text']).split()),
        'label': sample['label']
    }

# imdb 데이터셋을 사용한 영화리뷰 부정 확인
data = load_dataset('imdb')

preprocessed = data.map(preprocess)

# tokenizer
# AutoTokenizer 또한 존재하여 자동 Tokenizer 로드 가능
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

# tokenizer = 'input_ids', 'token_type_ids', 'attention_mask'
preprocessed = preprocessed.map(
    lambda sample: tokenizer(sample['text'], truncation=True),
    remove_columns=['text'],
    batched=True
)

collator = DataCollatorWithPadding(tokenizer)

# model 준비
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# GPU load
# model.cuda()
# pytorch가 아닌 huggingface에서 trainer를 사용
# 학습 방법을 명시
training_args = TrainingArguments(
    num_train_epochs=3.0,
    per_device_train_batch_size=16,
    output_dir='dump/test'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed['train'],
    eval_dataset=preprocessed['test'],
    data_collator=collator
)

trainer.train()
