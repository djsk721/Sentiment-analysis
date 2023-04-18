#%%
import torch
import evaluate 
import pandas as pd
import numpy as np
from Korpora import Korpora
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AlbertConfig, AutoModelForSequenceClassification


class DataloaderDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
NSMC = Korpora.load('nsmc')

train_data = pd.DataFrame({"texts":NSMC.train.texts, "labels":NSMC.train.labels})
test_data = pd.DataFrame({"texts":NSMC.test.texts, "labels":NSMC.test.labels})

pretrained_model_name="beomi/kcbert-base"
save_name = "trained_model"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name
)


tokenized_train_sentences = tokenizer(
    list(train_data.texts),
    return_tensors="pt",
    padding=True,
    truncation=True,
)

tokenized_test_sentences = tokenizer(
    list(test_data.texts),
    return_tensors="pt",
    padding=True,
    truncation=True
)

train_label = train_data['labels'].values
test_label = test_data['labels'].values

train_dataset = DataloaderDataset(tokenized_train_sentences, train_label)
test_dataset = DataloaderDataset(tokenized_test_sentences, test_label)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


pretrained_model_config = AlbertConfig.from_pretrained(
    pretrained_model_name,
)

model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        config=pretrained_model_config,
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    # per_device_train_batch_size=16,  # batch size per device during training
    # per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    save_steps=200,
    save_total_limit=2,
    save_on_each_node=True,
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",
    seed=3
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(save_name)
# %%
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoTokenizer
import torch
import numpy as np
import pandas as pd 


class DataloaderDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
    
pretrained_model_name="beomi/kcbert-base"
save_name = "trained_model"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name
)

test_data = pd.DataFrame({"texts":["이러한 내용을 지금까지 몰랐다니, 정말 다시 보고 싶다.",
                                   "오 좋은데","이상하다",
                                   "ㅋㅋ나는 재미있는지 잘모르겠다",
                                   "오랜만에 굉장히 훌륭한 만남"]})

tokenized_test_sentences = tokenizer(
    list(test_data.texts),
    return_tensors="pt",
    padding=True,
    truncation=True,
)

pred_dataset = DataloaderDataset(tokenized_test_sentences)

model_loaded = AutoModelForSequenceClassification.from_pretrained(save_name)
trainer = Trainer(model = save_name)
pred_results = trainer.predict(pred_dataset)
predictions = np.argmax(pred_results.predictions, axis=-1)
test_data["labels"]=predictions

