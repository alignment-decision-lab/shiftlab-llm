from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from datasets import Dataset
from datasets import load_dataset


MODEL_NAME = "gpt2"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # Padding makes sequences in the same batch have the same length.
    return tokenizer

def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model

def compute_loss(model, tokenizer, text):
    tokens = tokenizer(text)
    input_ids = torch.tensor([tokens["input_ids"]])
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    return outputs.loss

def build_dataset(texts):
    dataset = Dataset.from_dict({"text": texts})
    return dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True) #truncation ensures that the input does not exceed the maximum length allowed by the model.

def build_data_collator(tokenizer): # The data collator is responsible for collating the input sequences into batches and applying any necessary padding to ensure that all sequences in a batch have the same length. In this case, we are using the DataCollatorForLanguageModeling, which is designed for language modeling tasks and will pad the input sequences to the maximum length in the batch.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return data_collator

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_model()
    print("Tokenizer and model loaded successfully.")
    data_collator = build_data_collator(tokenizer)

    #text = "I love storms"
    #tokens = tokenizer(text)
    #print("Tokenized text:", tokens)
    #input_ids = torch.tensor([tokens["input_ids"]])
    #outputs = model(input_ids=input_ids, labels=input_ids) # the model will output the mean loss for the input sequence, which we can use to evaluate the model's performance on the given text.
    #print("Model output:", outputs)
    #print("Loss:", outputs.loss)

    #text1 = "I love storms."
    #text2 = "I really love storms !!!"
    #l1 = compute_loss(model, tokenizer, text1)
    #l2 = compute_loss(model, tokenizer, text2)
    #print("Loss for text1:", l1)
    #print("Loss for text2:", l2)

    #texts = [
    #    "Question: What is humidity? Answer: Humidity is the amount of water vapor in the air.",
    #    "Question: What is pressure? Answer: Pressure is force per unit area.",
    #    "Question: What is wind? Answer: Wind is air in motion.",
    #    "Question: What is temperature? Answer: Temperature measures how hot or cold something is.",
    #    "Question: What is a storm? Answer: A storm is a disturbed state of the atmosphere.",
    #   "Question: What is rain? Answer: Rain is liquid water falling from clouds."
    #]

    #dataset = build_dataset(texts)
    #tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched = True) # .map applies the function to each row in the dataset
    #print("Dataset:", dataset)
    #print("First example:", dataset[0])
    #print("Tokenized dataset:", tokenized_dataset)
    #print("First tokenized example:", tokenized_dataset[0])
    #split_dataset = tokenized_dataset.train_test_split(test_size=0.33, seed=42) #33% of the data will be used for validation.
    #train_dataset = split_dataset["train"]
    #val_dataset = split_dataset["test"]
    #print("Train dataset:", train_dataset)
    #print("Validation dataset:", val_dataset)
    #print("Data collator:", data_collator)

    #training_args = TrainingArguments(
    #output_dir="./results",
    #num_train_epochs=1,
    #per_device_train_batch_size=1,
    #per_device_eval_batch_size=1,
    #logging_steps=1,
    #report_to="none",
    #save_strategy="no",
    #use_cpu=True,
#)
    #trainer = Trainer(
    #model=model,
    #args=training_args,
    #train_dataset=train_dataset,
    #eval_dataset=val_dataset,
    #data_collator=data_collator,
#)
    #trainer.train()
    #eval_results = trainer.evaluate(eval_dataset=val_dataset)
    #print(eval_results)
    
    #dataset_l = load_dataset("wikitext", "wikitext-2-raw-v1")
    #Texts = dataset_l["train"]["text"][:50] #taking only the first 50 lines of the dataset for training.
    #Texts = [t for t in Texts if len(t.strip()) > 0] #removing empty lines from the dataset.
    #dataset_T = build_dataset(Texts)
    #tokenized_dataset_T = dataset_T.map(lambda x: tokenize_function(x, tokenizer), batched = True)
    #split_dataset_T = tokenized_dataset_T.train_test_split(test_size=0.33, seed=42)
    #train_dataset_T = split_dataset_T["train"]
    #val_dataset_T = split_dataset_T["test"]
    #trainer_T = Trainer(
    #model=model,
    #args=training_args,
    #train_dataset=train_dataset_T,
    #eval_dataset=val_dataset_T,
    #data_collator=data_collator,
#)
    #trainer_T.train()
    #eval_results_T = trainer_T.evaluate(eval_dataset=val_dataset_T)
    #print(eval_results_T)


    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.select(range(20)) #no need to modify the dataset since there is already a "text" column in the dataset that contains the input and output for each example.
    print(dataset[0])

    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched = True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.33, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
# Automatic training:
    #trainer = Trainer(
    #model=model,
    #args=training_args,
    #train_dataset=train_dataset,
    #eval_dataset=val_dataset,
    #data_collator=data_collator,
#)
    #trainer.train()
    #eval_results = trainer.evaluate(eval_dataset=val_dataset)
    #print(eval_results)

# Manual training:
train_loader = DataLoader(
    train_dataset,
    shuffle=True, # here we used shuffle but we could do sampling instead, which would allow us to use a weighted sampling strategy to balance the dataset if needed.
    batch_size=1,
    collate_fn=data_collator # makes sure that the input sequences are padded to the same lenght in each batch and has labels for the language modeling task.
) # Creates batches of data for training.
optimizer = optim.SGD(model.parameters(), lr=0.1) # the optimizer is stochastic gradient decsent
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
    val_dataset,
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

    #L=[]
    #for i in range(3):
    #    text = dataset[i]["text"]
    #    tokens = tokenizer(text)
    #    input_ids = torch.tensor([tokens["input_ids"]])
    #    outputs = model(input_ids=input_ids) # the model will output the logits for each token in the input sequence, which we can use to compute the loss for each token.
    #    logits = outputs.logits
    #    labels = input_ids
    #    logits = logits[:, :-1, :]
    #    labels = labels[:, 1:]
    #    logits = logits.reshape(-1, logits.size(-1)) # logits:(batch, sequence_length, vocab_size) -> (batch*sequence_length, vocab_size)
    #    labels = labels.reshape(-1) # labels:(batch, sequence_length) -> (batch*sequence_length)
    #    losses = F.cross_entropy(logits, labels, reduction="none") #reduction="none" means that we want to keep the loss for each token in the sequence, instead of averaging them.
    #    loss = losses.mean()
    #    L.append(loss.item())

#print("Loss per sentence:", L)