import tiny_financial_advisor.openai as openai
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI (you'll need to set OPENAI_API_KEY in your .env file)
openai.api_key = os.getenv('OPENAI_API_KEY')

class UserProfile(BaseModel):
    user_id: str
    income: float
    savings: float
    risk_tolerance: int
    investment_timeline: int
    financial_goals: List[str]
    existing_investments: Dict[str, float]

class InvestmentRecommendation(BaseModel):
    asset_allocation: Dict[str, float]
    recommended_products: List[str]
    expected_returns: float
    risk_level: int
    rationale: str

class FinancialAdvisorBot:
    def __init__(self):
        self.conversation_history = []
        self.user_profile = None
        self.market_data = self._initialize_market_data()
        
    def _initialize_market_data(self) -> Dict:
        """Initialize simple market data for demonstration"""
        return {
            "investment_products": {
                "stocks": {
                    "SP500_ETF": {"risk": 6, "expected_return": 0.10},
                    "NASDAQ_ETF": {"risk": 7, "expected_return": 0.12},
                    "DIV_ETF": {"risk": 5, "expected_return": 0.08}
                },
                "bonds": {
                    "GOVT_BOND_ETF": {"risk": 3, "expected_return": 0.04},
                    "CORP_BOND_ETF": {"risk": 4, "expected_return": 0.05}
                },
                "balanced": {
                    "TARGET_2030": {"risk": 5, "expected_return": 0.07},
                    "TARGET_2040": {"risk": 6, "expected_return": 0.08},
                    "TARGET_2050": {"risk": 7, "expected_return": 0.09}
                }
            }
        }

    async def chat(self, user_input: str) -> str:
        """Handle conversation with user"""
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Create system message based on conversation state
        if not self.user_profile:
            system_message = """You are a financial advisor AI. Collect the following information from the user:
            1. Annual income
            2. Current savings
            3. Risk tolerance (1-10)
            4. Investment timeline (in years)
            5. Financial goals
            6. Existing investments
            
            Be conversational and collect one piece of information at a time. Don't overwhelm the user."""
        else:
            system_message = """You are a financial advisor AI. You have collected the user's information.
            Provide investment advice based on their profile and respond to their questions."""

        # Get response from GPT
        response = await self._get_gpt_response(system_message)
        
        # Process the response
        if not self.user_profile:
            self._try_extract_user_data(response)
        
        return response

    async def _get_gpt_response(self, system_message: str) -> str:
        """Get response from GPT"""
        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history
        ]
        
        response = await openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": response_text})
        return response_text

    def _try_extract_user_data(self, response: str) -> None:
        """Try to extract user profile data from conversation"""
        try:
            # Use simple rules to extract numbers and text
            if "income" in response.lower() and any(char.isdigit() for char in response):
                income = float(''.join(filter(str.isdigit, response)))
                self._update_user_profile("income", income)
            
            if "savings" in response.lower() and any(char.isdigit() for char in response):
                savings = float(''.join(filter(str.isdigit, response)))
                self._update_user_profile("savings", savings)
            
            if "risk" in response.lower() and any(char.isdigit() for char in response):
                risk = int(''.join(filter(str.isdigit, response)))
                if 1 <= risk <= 10:
                    self._update_user_profile("risk_tolerance", risk)
            
            # Add more extraction rules as needed
        except Exception as e:
            print(f"Error extracting user data: {e}")

    def _update_user_profile(self, field: str, value: any) -> None:
        """Update user profile with new information"""
        if not self.user_profile:
            self.user_profile = UserProfile(
                user_id="user_1",
                income=0,
                savings=0,
                risk_tolerance=5,
                investment_timeline=10,
                financial_goals=[],
                existing_investments={}
            )
        
        setattr(self.user_profile, field, value)

    def generate_recommendation(self) -> InvestmentRecommendation:
        """Generate investment recommendations based on user profile"""
        if not self.user_profile:
            raise ValueError("User profile not complete")
        
        risk_score = self._calculate_risk_score()
        allocation = self._calculate_asset_allocation(risk_score)
        products = self._select_investment_products(allocation)
        expected_returns = self._calculate_expected_returns(products)
        
        return InvestmentRecommendation(
            asset_allocation=allocation,
            recommended_products=products,
            expected_returns=expected_returns,
            risk_level=risk_score,
            rationale=self._generate_rationale(allocation, products, expected_returns)
        )

    def _calculate_risk_score(self) -> int:
        """Calculate risk score based on user profile"""
        base_score = self.user_profile.risk_tolerance
        
        # Adjust for investment timeline
        timeline_factor = min(self.user_profile.investment_timeline / 10, 1)
        
        # Adjust for income/savings ratio
        income_savings_ratio = min(self.user_profile.savings / (self.user_profile.income + 1), 2)
        
        risk_score = base_score * 0.6 + timeline_factor * 2 + income_savings_ratio * 2
        return min(int(risk_score), 10)

    def _calculate_asset_allocation(self, risk_score: int) -> Dict[str, float]:
        """Calculate asset allocation based on risk score"""
        # Simple allocation strategy
        stocks_percentage = risk_score * 10
        bonds_percentage = 100 - stocks_percentage
        
        return {
            "stocks": stocks_percentage,
            "bonds": bonds_percentage
        }

    def _select_investment_products(self, allocation: Dict[str, float]) -> List[str]:
        """Select specific investment products based on allocation"""
        products = []
        
        if allocation["stocks"] > 60:
            products.append("SP500_ETF")
            products.append("NASDAQ_ETF")
        elif allocation["stocks"] > 30:
            products.append("SP500_ETF")
            products.append("DIV_ETF")
        
        if allocation["bonds"] > 40:
            products.append("GOVT_BOND_ETF")
            products.append("CORP_BOND_ETF")
        elif allocation["bonds"] > 0:
            products.append("GOVT_BOND_ETF")
        
        return products

    def _calculate_expected_returns(self, products: List[str]) -> float:
        """Calculate expected returns based on selected products"""
        total_return = 0
        weight = 1.0 / len(products)
        
        for product in products:
            for category in self.market_data["investment_products"].values():
                if product in category:
                    total_return += category[product]["expected_return"] * weight
        
        return round(total_return * 100, 2)

    def _generate_rationale(self, allocation: Dict[str, float], products: List[str], 
                          expected_returns: float) -> str:
        """Generate explanation for investment recommendations"""
        return f"""Based on your risk profile and investment timeline, we recommend:
        - {allocation['stocks']}% in stocks and {allocation['bonds']}% in bonds
        - Specific products: {', '.join(products)}
        - Expected annual return: {expected_returns}%
        
        This balanced approach aims to achieve your financial goals while managing risk."""

    def simulate_investment(self, recommendation: InvestmentRecommendation) -> bool:
        """Simulate executing the investment recommendation"""
        # In a real system, this would connect to a trading platform
        # For now, we'll just simulate success
        print(f"Simulating investment execution:")
        print(f"- Allocating assets: {recommendation.asset_allocation}")
        print(f"- Buying products: {recommendation.recommended_products}")
        return True

# Example usage
async def main():
    bot = FinancialAdvisorBot()
    
    # Simulate conversation
    prompts = [
        "Hi, I need help with investing my money.",
        "My annual income is $75,000",
        "I have $50,000 in savings",
        "My risk tolerance is 7 out of 10",
        "I want to invest for 15 years",
        "My goal is to save for retirement",
        "I don't have any existing investments"
    ]
    
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        response = await bot.chat(prompt)
        print(f"Bot: {response}")
    
    # Generate and show recommendation
    try:
        recommendation = bot.generate_recommendation()
        print("\nFinal Investment Recommendation:")
        print(json.dumps(recommendation.dict(), indent=2))
        
        # Simulate investment
        success = bot.simulate_investment(recommendation)
        if success:
            print("\nInvestment simulation completed successfully!")
    except ValueError as e:
        print(f"\nError: {e}")

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())