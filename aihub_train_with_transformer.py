#%%
import torch
import pandas as pd

#transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
# from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, DatasetDict
import wandb
import random


MODEL = "monologg/distilkobert"
LEARNING_RATE = 2e-4
EPOCHS = 10

# # start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Sentiment Analysis",
    name = f"{MODEL}_{LEARNING_RATE}_{EPOCHS}epochs",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": MODEL,
    "dataset": "AI-HUB 감성대화 말뭉치",
    "epochs": EPOCHS,
    }
)

#GPU 사용
device = torch.device("cuda")

sentiment_dataframe = pd.read_csv('sentiment_dialogues.csv')

sentiment_dataframe.loc[(sentiment_dataframe['label'] == "불안"), 'label'] = 0  #불안 => 0
sentiment_dataframe.loc[(sentiment_dataframe['label'] == "당황"), 'label'] = 1  #당황 => 1
sentiment_dataframe.loc[(sentiment_dataframe['label'] == "분노"), 'label'] = 2  #분노 => 2
sentiment_dataframe.loc[(sentiment_dataframe['label'] == "슬픔"), 'label'] = 3  #슬픔 => 3
sentiment_dataframe.loc[(sentiment_dataframe['label'] == "기쁨"), 'label'] = 4  #기쁨 => 4
sentiment_dataframe.loc[(sentiment_dataframe['label'] == "상처"), 'label'] = 5  #상처 => 5

id2label = {0: "불안", 1: "당황", 2: "분노", 3: "슬픔", 4: "기쁨", 5: "상처"}
label2id = {"불안": 0, "당황": 1, "분노": 2, "슬픔": 3, "기쁨": 4, "상처": 5}

# emotion_to_dict = sentiment_dataframe['label'].value_counts().to_dict()

# id2label = {}
# label2id = {}
# i = 0
# for key, value in emotion_to_dict.items():
#     id2label[key] = i
#     i += 1

# label2id = {v:k for k,v in id2label.items()}

# for key, value in id2label.items():
#     sentiment_dataframe.loc[(sentiment_dataframe['label'] == key), 'label'] = value

sentiment_dataset = Dataset.from_pandas(sentiment_dataframe[['text', 'label']])
# 90% train, 10% validation
train_testvalid = sentiment_dataset.train_test_split(test_size=0.1)

# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'valid': train_testvalid['test']
})


# %%
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = train_test_valid_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# %%
import evaluate
import numpy as np


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir= f"distilkobert_{LEARNING_RATE}_{EPOCHS}epochs",
    learning_rate= LEARNING_RATE,
    per_device_train_batch_size= 64 ,
    per_device_eval_batch_size= 64 ,
    num_train_epochs= EPOCHS,
    weight_decay = 0.01 ,
    evaluation_strategy= "epoch" ,
    save_strategy= "epoch" ,
    load_best_model_at_end= True,
    report_to="wandb"
)  
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized[ "train" ],
    eval_dataset=tokenized[ "valid" ],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics, 
)  
trainer.train()
wandb.finish()
# %%
# inference
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch 


PATH = 'result/distilkobert_2e-05_10epochs\checkpoint-8200'
MODEL = "monologg/distilkobert"
text = "오늘 친구 집에 가기로 했는데, 약속이 안정해져서 짜증나. 그래서 그냥 집에 왔어. 그래서 기분이 너무 우울해"

model = AutoModelForSequenceClassification.from_pretrained(PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    print(model.config.id2label[predicted_class_id])
# %%
