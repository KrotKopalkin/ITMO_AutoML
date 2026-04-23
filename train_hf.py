import pandas as pd
import numpy as np
import os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.data_loader import load_config, load_data
from src.utils import set_seed
import evaluate

def main():
    config = load_config()
    set_seed(config.get('seed', 42))
    
    train_df, test_df = load_data(config)
    
    # 1. Prepare Target
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['author'])
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}
    
    # 2. Train-Test Split
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), 
        test_size=0.1, 
        stratify=train_df['label'], 
        random_state=42
    )
    
    ds_train = Dataset.from_pandas(train_df.iloc[train_idx][['text', 'label']])
    ds_val = Dataset.from_pandas(train_df.iloc[val_idx][['text', 'label']])
    
    # 3. Tokenization
    model_name = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
    
    tokenized_train = ds_train.map(tokenize_function, batched=True)
    tokenized_val = ds_val.map(tokenize_function, batched=True)
    
    # 4. Metrics
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
        loss = log_loss(labels, probs)
        predictions = np.argmax(logits, axis=-1)
        acc = metric.compute(predictions=predictions, references=labels)
        return {"accuracy": acc["accuracy"], "log_loss": loss}

    # 5. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir="./hf_results",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        fp16=True,
        report_to="tensorboard"
    )
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    print("Starting Fine-tuning of E5-multilingual...")
    trainer.train()
    
    # 8. Evaluation
    eval_results = trainer.evaluate()
    print(f"\nHF Model Validation Results: {eval_results}")
    
    # 9. Test Prediction
    ds_test = Dataset.from_pandas(test_df[['text']])
    tokenized_test = ds_test.map(tokenize_function, batched=True)
    
    print("Predicting on test set...")
    test_results = trainer.predict(tokenized_test)
    test_probs = torch.nn.functional.softmax(torch.from_numpy(test_results.predictions), dim=-1).numpy()
    
    # Prepare submission
    submission = pd.DataFrame(test_probs, columns=le.classes_)
    submission.insert(0, 'id', test_df['id'])
    
    output_path = os.path.join(config['paths']['output_dir'], "submission_hf_e5.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()
