import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json

# Sample training data - you should expand this
TRAINING_DATA = [
    {
        "conversation": [
            {"role": "human", "content": "I make 80,000 dollars per year and have 20,000 in savings. I want to invest for retirement."},
            {"role": "assistant", "content": "Thank you for sharing your financial information. Based on your income of $80,000 and savings of $20,000, let's discuss your retirement investment strategy. First, I'd like to understand your risk tolerance on a scale of 1-10, and how many years until you plan to retire?"}
        ]
    },
    {
        "conversation": [
            {"role": "human", "content": "My risk tolerance is 7 and I want to retire in 25 years"},
            {"role": "assistant", "content": "Given your relatively high risk tolerance of 7/10 and long investment horizon of 25 years, we can consider a more aggressive growth strategy. Would you be comfortable with a portfolio that's 80% stocks and 20% bonds? This could target higher long-term returns while maintaining some stability through bonds."}
        ]
    },
    {
        "conversation": [
            {"role": "human", "content": "I earn 10 lakhs per year and have 5 lakhs in savings"},
            {"role": "assistant", "content": "I understand you have an annual income of ₹1,000,000 (10 lakhs) and savings of ₹500,000 (5 lakhs). That's a good foundation. To help create an investment strategy, could you tell me your risk tolerance on a scale of 1-10, and what your primary financial goals are?"}
        ]
    },
    # Add more training examples here
]

def prepare_training_data(data):
    """Convert conversations into training format"""
    formatted_data = []
    
    for item in data:
        conversation = ""
        for message in item["conversation"]:
            if message["role"] == "human":
                conversation += f"Human: {message['content']}\n"
            else:
                conversation += f"Assistant: {message['content']}\n"
        formatted_data.append({"text": conversation})
    
    return formatted_data

def create_dataset():
    """Create dataset from training data"""
    formatted_data = prepare_training_data(TRAINING_DATA)
    return Dataset.from_list(formatted_data)

def train_model(base_model="mosaicml/mpt-7b", output_dir="financial_advisor_model"):
    """Fine-tune the model"""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare dataset
    dataset = create_dataset()
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        save_strategy="epoch",
        logging_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

def load_and_use_model(model_path="financial_advisor_model"):
    """Load and use the fine-tuned model"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    def generate_response(user_input):
        # Format input
        prompt = f"Human: {user_input}\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and return
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()

    return generate_response

# Example usage
def main():
    # First time: Train the model
    print("Training model...")
    train_model()
    
    # Load and use the model
    print("Loading model...")
    generate_response = load_and_use_model()
    
    print("Financial Advisor Bot is ready!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        response = generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()