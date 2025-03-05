import pandas as pd
import json
from datasets import load_dataset, Dataset
import requests
from typing import List, Dict
import os
from tqdm import tqdm

def get_financial_datasets():
    """Collect and combine multiple financial advice datasets"""
    datasets = []

    # 1. Load FinQA dataset - contains financial QA pairs
    print("Loading FinQA dataset...")
    try:
        finqa = load_dataset("financial_qa", split="train")
        # Convert to our format
        for item in finqa:
            datasets.append({
                "instruction": "You are a financial advisor. Answer the following financial question.",
                "input": item["question"],
                "output": item["answer"]
            })
    except Exception as e:
        print(f"Note: Could not load FinQA dataset: {e}")

    # 2. Load Personal Finance Subreddit Data
    # You can download this from: https://files.pushshift.io/reddit/
    print("Loading Reddit Personal Finance data...")
    reddit_data_path = "personal_finance_qa.json"
    if os.path.exists(reddit_data_path):
        try:
            with open(reddit_data_path, 'r') as f:
                reddit_data = json.load(f)
                for qa in reddit_data[:1000]:  # Limit to first 1000 high-quality QA pairs
                    datasets.append({
                        "instruction": "You are a financial advisor. Provide advice for the following financial situation.",
                        "input": qa["question"],
                        "output": qa["answer"]
                    })
        except Exception as e:
            print(f"Error loading Reddit data: {e}")

    # 3. Load Financial Professional Advice Dataset
    # This is a curated dataset of professional financial advisor responses
    print("Loading Professional Financial Advice data...")
    professional_data_path = "professional_finance_qa.json"
    if os.path.exists(professional_data_path):
        try:
            with open(professional_data_path, 'r') as f:
                prof_data = json.load(f)
                datasets.extend(prof_data)
        except Exception as e:
            print(f"Error loading professional data: {e}")

    # 4. Generate structured financial QA pairs
    print("Generating structured financial QA pairs...")
    datasets.extend(generate_structured_qa())

    return datasets

def generate_structured_qa() -> List[Dict]:
    """Generate structured QA pairs for common financial scenarios"""
    
    # Template data for generating QA pairs
    income_ranges = [
        "50,000", "75,000", "100,000", "10 lakhs", "20 lakhs",
        "£30,000", "£50,000", "€40,000"
    ]
    
    savings_ranges = [
        "10,000", "25,000", "50,000", "5 lakhs", "10 lakhs",
        "£10,000", "£20,000", "€15,000"
    ]
    
    risk_levels = ["conservative (3/10)", "moderate (5/10)", "aggressive (8/10)"]
    
    timeframes = ["5 years", "10 years", "20 years"]
    
    goals = [
        "retirement", "buying a house", "children's education", 
        "starting a business"
    ]

    qa_pairs = []

    # Generate sample combinations (not all to keep dataset size manageable)
    sample_count = 0
    max_samples = 50  # Limit generated samples for practicality
    
    for income in income_ranges[:3]:
        for savings in savings_ranges[:3]:
            for risk in risk_levels:
                for timeframe in timeframes[:2]:
                    for goal in goals[:2]:
                        if sample_count >= max_samples:
                            break
                            
                        # Basic financial profile question
                        input_text = f"I earn {income} annually and have {savings} in savings. "\
                                   f"My risk tolerance is {risk} and I want to invest for {timeframe} "\
                                   f"for {goal}."
                        
                        # Generate appropriate response based on parameters
                        output_text = generate_financial_advice(
                            income, savings, risk, timeframe, goal
                        )
                        
                        qa_pairs.append({
                            "instruction": "You are a financial advisor. Provide personalized investment advice based on the user's financial situation.",
                            "input": input_text,
                            "output": output_text
                        })
                        
                        sample_count += 1
                    if sample_count >= max_samples:
                        break
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
        if sample_count >= max_samples:
            break

    return qa_pairs

def generate_financial_advice(income: str, savings: str, risk: str, timeframe: str, goal: str) -> str:
    """Generate appropriate financial advice based on parameters"""
    
    # Extract risk level number
    risk_level = int(risk.split('/')[0][-1])
    
    # Parse timeframe
    years = int(timeframe.split()[0])
    
    # Basic portfolio allocation based on risk and timeframe
    if risk_level <= 4:  # Conservative
        stocks = 40
        bonds = 50
        cash = 10
    elif risk_level <= 7:  # Moderate
        stocks = 60
        bonds = 30
        cash = 10
    else:  # Aggressive
        stocks = 80
        bonds = 15
        cash = 5

    # Adjust for timeframe
    if years < 10:
        stocks = max(stocks - 10, 30)
        bonds = min(bonds + 10, 60)
    
    advice = f"""Based on your financial profile, here's my recommended investment strategy:

Portfolio Allocation:
- {stocks}% Stocks/Equity Funds
- {bonds}% Bonds/Fixed Income
- {cash}% Cash/Liquid Funds

Specific Recommendations:
1. First ensure you have an emergency fund of 6 months' expenses
2. Maximize any available tax-advantaged accounts
3. For stocks, consider a mix of:
   - Index funds for core exposure
   - Thematic funds aligned with {goal}
   - Geographic diversification

Risk Management:
- Regular portfolio rebalancing
- Dollar-cost averaging for investments
- Insurance coverage review

This allocation balances your {risk} risk tolerance with your {timeframe} investment horizon for {goal}."""

    return advice

def prepare_training_data():
    """Prepare the combined dataset for training"""
    
    # Get datasets
    print("Collecting datasets...")
    datasets = get_financial_datasets()
    
    # Format for training
    formatted_data = []
    for item in datasets:
        text = f"<|system|>{item['instruction']}<|user|>{item['input']}<|assistant|>{item['output']}"
        formatted_data.append({"text": text})
    
    # Create and save the dataset
    dataset = Dataset.from_list(formatted_data)
    dataset.save_to_disk("financial_advisor_dataset")
    print(f"Dataset saved with {len(dataset)} examples")
    
    return dataset

if __name__ == "__main__":
    # If run directly, prepare and save the dataset
    dataset = prepare_training_data()
    print(f"Total examples in dataset: {len(dataset)}")
    
    # Display some examples
    print("\nSample entries:")
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(dataset[i]['text'])