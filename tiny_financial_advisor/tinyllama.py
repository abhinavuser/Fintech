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
import os

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
OUTPUT_DIR = "tiny_financial_advisor"

# Focused financial training data
TRAINING_DATA = [
    {
        "instruction": "You are a financial advisor. Give appropriate investment advice based on the user's financial situation.",
        "input": "I make 10 lakhs per year and have 5 lakhs saved.",
        "output": "I understand you have an annual income of ₹1,000,000 (10 lakhs) and savings of ₹500,000 (5 lakhs). To provide appropriate investment advice, I need to know:\n1. Your risk tolerance (1-10 scale)\n2. Your investment timeline\n3. Your financial goals\n\nCould you share these details?"
    },
    {
        "instruction": "You are a financial advisor. Provide investment allocation advice based on risk tolerance.",
        "input": "I have a risk tolerance of 7/10 and want to invest for 20 years for retirement.",
        "output": "Given your high risk tolerance (7/10) and long investment horizon (20 years), I recommend:\n- 75% in equity funds (diversified across large, mid, and small cap)\n- 15% in bonds for stability\n- 10% in other assets like gold or REITs\n\nWould you like specific investment options for each category?"
    },
    {
        "instruction": "You are a financial advisor. Help identify appropriate investment products.",
        "input": "I want safe investments with regular income. I have low risk tolerance.",
        "output": "For safe, income-generating investments, I recommend:\n1. Government bonds and treasury bills\n2. Fixed deposits with reputable banks\n3. High-rated corporate bonds\n4. Monthly income schemes\n\nThese typically yield 5-7% annually with minimal risk. How much would you like to invest?"
    }
    # Add more examples as needed
]

def prepare_training_data():
    """Prepare data in instruction-tuning format"""
    formatted_data = []
    
    for item in TRAINING_DATA:
        # Format: <|system|>instruction<|user|>input<|assistant|>output
        text = f"<|system|>{item['instruction']}<|user|>{item['input']}<|assistant|>{item['output']}"
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def train_model():
    """Fine-tune the tiny model"""
    print(f"Loading {MODEL_NAME}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Prepare dataset
    dataset = prepare_training_data()
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments optimized for small model
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        evaluation_strategy="epoch",  # <-- Fix: Match eval and save strategy
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=1,
)

    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the model with minimal footprint
    model.save_pretrained(OUTPUT_DIR, save_in_half_precision=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Clean up checkpoints to save space
    for file in os.listdir(OUTPUT_DIR):
        if 'checkpoint' in file:
            checkpoint_dir = os.path.join(OUTPUT_DIR, file)
            if os.path.isdir(checkpoint_dir):
                for subfile in os.listdir(checkpoint_dir):
                    os.remove(os.path.join(checkpoint_dir, subfile))
                os.rmdir(checkpoint_dir)

class FinancialAdvisorBot:
    def __init__(self, model_path=OUTPUT_DIR):
        """Initialize the bot with the fine-tuned model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.conversation_history = []

    def get_response(self, user_input: str) -> str:
        """Generate response based on user input and conversation history"""
        # Format the conversation history and current input
        context = ""
        for msg in self.conversation_history[-3:]:  # Keep last 3 messages for context
            context += f"<|{msg['role']}|>{msg['content']}"
        
        prompt = f"{context}<|user|>{user_input}<|assistant|>"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

def main():
    """Main function to run the bot"""
    # Train model if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        train_model()
    
    # Initialize bot
    bot = FinancialAdvisorBot()
    
    print("Financial Advisor Bot: Hi! I'm your AI financial advisor. How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Financial Advisor Bot: Goodbye! Feel free to return for more financial advice.")
            break
            
        response = bot.get_response(user_input)
        print(f"Financial Advisor Bot: {response}")

if __name__ == "__main__":
    main()