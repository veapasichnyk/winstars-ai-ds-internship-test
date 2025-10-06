import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

# Default config
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-5
OUTPUT_DIR = "task2_ner_image_pipeline/ner_model/ner_model_out"


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """Helper to align NER labels with tokens after tokenization."""
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_model(
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    output_dir: str = OUTPUT_DIR,
    max_train_samples: int = 5000,
):
    """
    Train a NER model on the English split of Babelscape/wikineural.
    """

    # Load dataset (English only)
    dataset = load_dataset("Babelscape/wikineural")
    train_dataset = dataset["train_en"]
    val_dataset = dataset["val_en"]

    # Optional: take only a subset for quick training
    if max_train_samples and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and align labels
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    # Build label list manually
    unique_labels = sorted(set(sum(train_dataset["ner_tags"], [])))
    num_labels = len(unique_labels)
    
    model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training args
    training_args = TrainingArguments(
      output_dir=output_dir,
      do_eval=True,
      eval_steps=500,
      learning_rate=lr,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      num_train_epochs=epochs,
      weight_decay=0.01,
      logging_dir="./logs",
      logging_steps=50,
      save_steps=500,
      optim="adamw_torch",
      )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train & save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f" NER model trained and saved to: {output_dir}")


if __name__ == "__main__":
    train_model()