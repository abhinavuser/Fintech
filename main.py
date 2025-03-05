import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json
import os
import pandas as pd
from tqdm import tqdm
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the dataset generation function from the first file
from financial_data_prep import prepare_training_data as load_financial_dataset

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
OUTPUT_DIR = "tiny_financial_advisor"

def train_model():
    """Fine-tune the tiny model using the comprehensive financial dataset"""
    try:
        print(f"Loading {MODEL_NAME}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Memory optimization for model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True  # Added for memory optimization
        )
        
        # Load the financial dataset
        print("Loading financial training data...")
        try:
            if os.path.exists("financial_advisor_dataset"):
                dataset = load_from_disk("financial_advisor_dataset")
                print(f"Loaded {len(dataset)} examples from saved dataset")
            else:
                print("Saved dataset not found. Generating new dataset...")
                dataset = load_financial_dataset()
                print(f"Generated {len(dataset)} training examples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to minimal dataset...")
            dataset = Dataset.from_list([
                {"text": "<|system|>You are a financial advisor. Provide investment advice.<|user|>Should I invest in stocks or bonds?<|assistant|>That depends on your risk tolerance and investment timeline. Stocks offer higher potential returns but with more volatility, while bonds provide more stable but typically lower returns."}
            ])
        
        # Ensure proper dataset format
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        elif isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset)

        def tokenize_function(examples):
            """Tokenization with memory optimization"""
            result = tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,  # Reduced for memory
                padding=False,
                return_tensors=None
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=1,  # Reduced batch size
            desc="Tokenizing dataset"
        )
        print(f"First example: {dataset[0]}")
        
        # Memory-optimized training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            fp16=torch.cuda.is_available(),
            save_strategy="epoch",
            logging_steps=1,  # More frequent logging
            save_total_limit=1,
            remove_unused_columns=True,
            prediction_loss_only=True,
            max_grad_norm=0.5,  # Added gradient clipping
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            gradient_checkpointing=True,  # Memory optimization
        )
        
        # Initialize trainer with error catching
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )
        
        print("Starting training...")
        try:
            trainer.train()
        except Exception as e:
            print(f"Training error occurred: {str(e)}")
            print("Detailed error info:", sys.exc_info())
            raise
        
        # Save the model
        print(f"Saving model to {OUTPUT_DIR}...")
        model.save_pretrained(OUTPUT_DIR, save_in_half_precision=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Clean up checkpoints
        for file in os.listdir(OUTPUT_DIR):
            if 'checkpoint' in file:
                checkpoint_dir = os.path.join(OUTPUT_DIR, file)
                if os.path.isdir(checkpoint_dir):
                    for subfile in os.listdir(checkpoint_dir):
                        os.remove(os.path.join(checkpoint_dir, subfile))
                    os.rmdir(checkpoint_dir)
                    
    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
        print("Detailed error info:", sys.exc_info())
        raise

    
    # Save the model with minimal footprint
    print(f"Saving model to {OUTPUT_DIR}...")
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
        # For inference, we can use device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.conversation_history = []
        # Add system prompt for financial advisor context
        self.conversation_history.append({
            "role": "system", 
            "content": "You are a skilled financial advisor providing personalized investment advice based on user needs."
        })

    def get_response(self, user_input: str) -> str:
        """Generate response based on user input and conversation history"""
        # Format the conversation history and current input
        context = ""
        for msg in self.conversation_history:  # Include all history for better context
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
    if not os.path.exists(OUTPUT_DIR) or not os.path.exists(os.path.join(OUTPUT_DIR, "pytorch_model.bin")):
        print(f"Model not found in {OUTPUT_DIR}. Starting training process...")
        train_model()
    else:
        print(f"Using existing model from {OUTPUT_DIR}")
    
    # Initialize bot
    bot = FinancialAdvisorBot()
    
    print("\n" + "="*50)
    print("Financial Advisor Bot: Hi! I'm your AI financial advisor. How can I help you today?")
    print("="*50 + "\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Financial Advisor Bot: Goodbye! Feel free to return for more financial advice.")
            break
            
        response = bot.get_response(user_input)
        print(f"Financial Advisor Bot: {response}")

if __name__ == "__main__":
    main()