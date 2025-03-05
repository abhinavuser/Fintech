from typing import Dict, List, Optional
from pydantic import BaseModel
import numpy as np
import json
import re

class UserProfile(BaseModel):
    user_id: str
    income: float
    savings: float
    risk_tolerance: int  # 1-10
    investment_timeline: int  # in years
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
        self.questions = [
            "What is your annual income?",
            "How much do you have in savings?",
            "On a scale of 1-10, how much risk are you willing to take? (1 = very conservative, 10 = very aggressive)",
            "How many years do you plan to invest for?",
            "What are your financial goals? (e.g., retirement, house, education)",
            "Do you have any existing investments? Please list them if you do."
        ]
        self.current_question = 0
        self.user_profile = None
        self.market_data = self._initialize_market_data()

    def _initialize_market_data(self) -> Dict:
        """Initialize market data with sample investment products"""
        return {
            "investment_products": {
                "stocks": {
                    "Total_Market_ETF": {"risk": 6, "expected_return": 0.10},
                    "Growth_ETF": {"risk": 7, "expected_return": 0.12},
                    "Dividend_ETF": {"risk": 5, "expected_return": 0.08}
                },
                "bonds": {
                    "Government_Bond_ETF": {"risk": 3, "expected_return": 0.04},
                    "Corporate_Bond_ETF": {"risk": 4, "expected_return": 0.05}
                },
                "balanced": {
                    "Target_2030": {"risk": 5, "expected_return": 0.07},
                    "Target_2040": {"risk": 6, "expected_return": 0.08},
                    "Target_2050": {"risk": 7, "expected_return": 0.09}
                }
            }
        }

    def chat(self, user_input: str) -> str:
        """Process user input and return appropriate response"""
        if not self.user_profile:
            # Still collecting initial information
            response = self._process_user_input(user_input)
            if self.current_question < len(self.questions):
                response += f"\n\n{self.questions[self.current_question]}"
            else:
                # Generate initial recommendation
                recommendation = self.generate_recommendation()
                response += f"\n\nBased on your information, here's my recommendation:\n{recommendation.rationale}"
            return response
        else:
            # Handle ongoing conversation
            return self._handle_ongoing_conversation(user_input)

    def _process_user_input(self, user_input: str) -> str:
        """Process user input during the initial questionnaire phase"""
        if self.current_question == 0:  # Income
            income = self._extract_number(user_input)
            if income:
                self._update_user_profile("income", income)
                self.current_question += 1
                return f"Got it! Your annual income is ${income:,.2f}."
            return "I didn't catch the income amount. Please provide a number."

        elif self.current_question == 1:  # Savings
            savings = self._extract_number(user_input)
            if savings:
                self._update_user_profile("savings", savings)
                self.current_question += 1
                return f"Great! You have ${savings:,.2f} in savings."
            return "I didn't catch the savings amount. Please provide a number."

        elif self.current_question == 2:  # Risk tolerance
            risk = self._extract_number(user_input)
            if risk and 1 <= risk <= 10:
                self._update_user_profile("risk_tolerance", int(risk))
                self.current_question += 1
                return f"Understood! Your risk tolerance is {int(risk)}/10."
            return "Please provide a number between 1 and 10 for your risk tolerance."

        elif self.current_question == 3:  # Investment timeline
            timeline = self._extract_number(user_input)
            if timeline:
                self._update_user_profile("investment_timeline", int(timeline))
                self.current_question += 1
                return f"Got it! You're planning to invest for {int(timeline)} years."
            return "Please provide the number of years you plan to invest for."

        elif self.current_question == 4:  # Financial goals
            goals = [goal.strip() for goal in user_input.split(',')]
            self._update_user_profile("financial_goals", goals)
            self.current_question += 1
            return f"Great! Your financial goals are: {', '.join(goals)}."

        elif self.current_question == 5:  # Existing investments
            investments = {}
            if "no" not in user_input.lower() and "don't" not in user_input.lower():
                # Simple parsing of investment text
                parts = user_input.split(',')
                for part in parts:
                    if ':' in part:
                        inv_type, amount = part.split(':')
                        investments[inv_type.strip()] = self._extract_number(amount) or 0
            self._update_user_profile("existing_investments", investments)
            self.current_question += 1
            return "Thanks for sharing your investment information!"

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text, handling currency and comma formatting"""
        text = text.replace('$', '').replace(',', '')
        numbers = re.findall(r'\d+\.?\d*', text)
        return float(numbers[0]) if numbers else None

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

    def _handle_ongoing_conversation(self, user_input: str) -> str:
        """Handle conversation after initial profile is complete"""
        # Simple keyword-based responses
        user_input = user_input.lower()
        if "risk" in user_input:
            return f"Your current risk tolerance is {self.user_profile.risk_tolerance}/10. Would you like to adjust it?"
        elif "return" in user_input or "performance" in user_input:
            recommendation = self.generate_recommendation()
            return f"Based on your profile, the expected annual return is {recommendation.expected_returns}%"
        elif "allocation" in user_input:
            recommendation = self.generate_recommendation()
            return f"Your recommended asset allocation is: {json.dumps(recommendation.asset_allocation, indent=2)}"
        else:
            return "I can help you with information about your risk level, expected returns, or asset allocation. What would you like to know?"

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
        timeline_factor = min(self.user_profile.investment_timeline / 10, 1)
        income_savings_ratio = min(self.user_profile.savings / (self.user_profile.income + 1), 2)
        risk_score = base_score * 0.6 + timeline_factor * 2 + income_savings_ratio * 2
        return min(int(risk_score), 10)

    def _calculate_asset_allocation(self, risk_score: int) -> Dict[str, float]:
        """Calculate asset allocation based on risk score"""
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
            products.extend(["Total_Market_ETF", "Growth_ETF"])
        elif allocation["stocks"] > 30:
            products.extend(["Total_Market_ETF", "Dividend_ETF"])
        
        if allocation["bonds"] > 40:
            products.extend(["Government_Bond_ETF", "Corporate_Bond_ETF"])
        elif allocation["bonds"] > 0:
            products.append("Government_Bond_ETF")
        
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
        - Recommended products: {', '.join(products)}
        - Expected annual return: {expected_returns}%
        
        This allocation is designed to balance your growth objectives with your risk tolerance."""

def main():
    """Main function to run the financial advisor bot"""
    bot = FinancialAdvisorBot()
    print("Financial Advisor Bot: Hi! I'll help you make investment decisions. Let's start with some questions.\n")
    print(bot.questions[0])

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nFinancial Advisor Bot: Thanks for chatting! Good bye!")
            break
            
        response = bot.chat(user_input)
        print(f"\nFinancial Advisor Bot: {response}")

if __name__ == "__main__":
    main()