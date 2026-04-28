# Dataset preparation:
from datasets import load_dataset

dataset = load_dataset(
    "bigscience-historical-texts/HIPE2020_sent-split",
    "fr",
)
#print(dataset)
#print(dataset.keys())
#print(dataset["train"][0])

def rebuild_text(example):
    tokens = example["tokens"] # "tokens" in the dataset of HIPE is a list of words/ponctuation that make up the text of the example.
    no_space_after = example["no_space_after"] # "no_space_after" in the dataset of HIPE is a list of booleans that indicate wether there is a space after the corresponding token(False) or not (True).

    text = ""
    for token, no_space in zip(tokens, no_space_after):
        text += token
        if not no_space: # means there is a space after the token, so we add a space to the text.
            text += " "

    example["text"] = text.strip() # .strip() removes spaces at the beginning and end of the text AND creates a new key "text" in the example that contains the reconstructed text of the example.
    return example

dataset = dataset.map(rebuild_text)

#print(dataset["train"][0]["text"])
    # Now we need to keep only the "text" column of the dataset:
columns_to_remove = [
    col for col in dataset["train"].column_names
    if col != "text"
]

dataset = dataset.remove_columns(columns_to_remove)

#print(dataset)
print(dataset["train"][0])

train_dataset = dataset["train"].select(range(20))
val_dataset =  dataset["validation"].select(range(5)) 

# GPT-2 Training:
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

MODEL_NAME = "gpt2"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # Padding makes sequences in the same batch have the same length.
    return tokenizer

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True) #truncation ensures that the input does not exceed the maximum length allowed by the model.

def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model

def build_data_collator(tokenizer): # The data collator is responsible for collating the input sequences into batches and applying any necessary padding to ensure that all sequences in a batch have the same length. In this case, we are using the DataCollatorForLanguageModeling, which is designed for language modeling tasks and will pad the input sequences to the maximum length in the batch.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return data_collator

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_model()
    print("Tokenizer and model loaded successfully.")
    data_collator = build_data_collator(tokenizer)

    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched = True, remove_columns=["text"]) # We remove the "text" column from the dataset since we don't need it anymore after tokenization, and we want to keep only the input_ids and attention_mask columns that are needed for training the model. And also it doesn't work when we keep it.
    tokenized_val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched = True, remove_columns=["text"])

    # Manual training:
    train_loader = DataLoader(
        tokenized_train_dataset,
        shuffle=True, # here we used shuffle but we could do sampling instead, which would allow us to use a weighted sampling strategy to balance the dataset if needed.
        batch_size=1,
        collate_fn=data_collator # makes sure that the input sequences are padded to the same lenght in each batch and has labels for the language modeling task.
) # Creates batches of data for training.
    optimizer = optim.SGD(model.parameters(), lr=0.001) # the optimizer is stochastic gradient decsent
    model.train() # sets the model to training mode.
    n_epochs = 1
    for epoch in range(n_epochs):
        for batch in train_loader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])# examples go through the model.
            loss = outputs.loss # computes the loss for the batch.
            optimizer.zero_grad() # resets the gradients for the model parameters.
            loss.backward() # computes the gradients for the model parameters.
            optimizer.step() # updates the model parameters based on the computed gradients.
            print("Training loss:", loss.item())
    val_loader = DataLoader(
        tokenized_val_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=data_collator
)# Creates batches of data for evaluation.
    model.eval() # sets the model to evaluation mode.
    Losses = []
    for batch in val_loader:
        with torch.no_grad(): # disables gradient computation since we are only evaluating the model and not updating its parameters.
            outputs = model(**batch)
            loss = outputs.loss
            Losses.append(loss.item())
    print("Mean validation loss:", sum(Losses)/len(Losses))
