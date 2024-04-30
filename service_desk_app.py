import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

# Function to generate response using GPT-2 model
def generate_response(user_query, model, tokenizer):
    input_text = "User: " + user_query + " Bot:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Function to fine-tune the model with updated training data
def fine_tune_model(model, tokenizer, train_data):
    # Fine-tune the model
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
    )

    trainer.train()

# Load fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialize training data
train_data = []

# Streamlit UI
st.title("IT Service Desk AI Assistant")

# User input field
user_query = st.text_input("You:")

if user_query.strip() != "":
    # Add user query to training data
    train_data.append({"input_ids": tokenizer.encode("User: " + user_query + " Bot:", return_tensors="pt")})

    # Fine-tune the model with updated training data
    fine_tune_model(model, tokenizer, train_data)

    # Generate response
    response = generate_response(user_query, model, tokenizer)

    # Display response
    st.write("Bot:", response)
