from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
import json
import re
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.database.database_manager import DatabaseManager
import yfinance as yf
import pandas as pd

class FinanceAgent:
    def __init__(self):
        self.db = DatabaseManager()
        self.setup_llm()
        self.setup_prompts()
        self._pending_operation = None
        self._chat_history = []
        self._current_user = None

    def setup_llm(self):
        """Setup the LLM with appropriate parameters."""
        self.llm = Ollama(
            model="llama3.1:8b-instruct-q4_0",
            temperature=0.7,
            base_url="http://localhost:11434"
        )

    def setup_prompts(self):
        """Setup conversation prompts with enhanced context."""
        self.prompt = PromptTemplate(
            input_variables=["query", "current_time", "user_data", "market_data", "chat_history"],
            template="""You are an advanced AI financial assistant named FinanceGPT. You help users manage their investments, 
            execute trades, and provide financial advice. You have access to real-time market data and user portfolios.

            Current Time: {current_time}
            User Information: {user_data}
            Market Data: {market_data}
            Recent Conversation: {chat_history}

            User Query: {query}

            Your capabilities include:
            1. Natural Conversation:
               - Discuss market trends, investment strategies
               - Explain financial concepts
               - Provide personalized advice based on portfolio

            2. Account Management:
               - Create/manage user accounts
               - Show account balance and portfolio
               - Display transaction history
               - Add/remove stocks from watchlist

            3. Trading Operations:
               - Execute stock trades (buy/sell)
               - Monitor positions
               - Set price alerts
               - Analyze potential trades

            4. Market Analysis:
               - Provide real-time quotes
               - Show technical indicators
               - Discuss market news
               - Compare stocks

            When handling trades or sensitive operations:
            - ALWAYS ask for final confirmation
            - Verify account balance for purchases
            - Check existing holdings for sales
            - Show relevant market data before trades

            When responding, format trades and operations as JSON with this structure:
            {{
                "type": "conversation|account|trade|analysis",
                "operation": "CREATE|READ|UPDATE|DELETE|BUY|SELL|ANALYZE",
                "data": {{
                    "symbol": "STOCK_SYMBOL",
                    "shares": NUMBER_OF_SHARES,
                    "price": CURRENT_PRICE
                }},
                "natural_response": "Your friendly response",
                "requires_confirmation": true,
                "show_data": true
            }}

            For casual conversation, respond naturally without JSON.
            Always maintain a professional yet friendly tone.
            """
        )

        self.chain = (
            self.prompt 
            | self.llm 
            | StrOutputParser()
        )

    def get_market_data(self, symbols: List[str] = None) -> Dict:
        """Fetch current market data for relevant symbols."""
        try:
            market_data = {
                "market_status": "OPEN",  # Simplified for testing
                "quotes": {}
            }
            
            # Only fetch required symbols
            if symbols:
                for symbol in symbols:
                    try:
                        quote = self.db.get_real_time_quote(symbol)
                        market_data["quotes"][symbol] = quote
                    except Exception as e:
                        print(f"Error fetching {symbol}: {e}")
            
            return market_data
        except Exception as e:
            print(f"Error getting market data: {e}")
            return {"market_status": "ERROR", "quotes": {}}

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze if text indicates buying, selling, or general inquiry."""
        text = text.lower()
        
        # Common patterns for trading intentions
        buy_patterns = ['buy', 'purchase', 'invest in', 'get some', 'acquire']
        sell_patterns = ['sell', 'dump', 'get rid of', 'dispose', 'exit']
        
        for pattern in buy_patterns:
            if pattern in text:
                # Extract stock symbol - basic pattern matching
                symbols = re.findall(r'[A-Z]{1,5}', text.upper())
                if symbols:
                    return {"action": "BUY", "symbol": symbols[0]}
        
        for pattern in sell_patterns:
            if pattern in text:
                symbols = re.findall(r'[A-Z]{1,5}', text.upper())
                if symbols:
                    return {"action": "SELL", "symbol": symbols[0]}
        
        return {"action": "ANALYZE"}

    def _format_portfolio_summary(self, portfolio: List[Dict]) -> str:
        """Format portfolio data for display."""
        if not portfolio:
            return "Your portfolio is empty."
        
        total_value = sum(pos['market_value'] for pos in portfolio)
        total_pl = sum(pos['profit_loss'] for pos in portfolio)
        
        summary = [
            "📊 Portfolio Summary:",
            f"Total Value: ${total_value:,.2f}",
            f"Total P/L: ${total_pl:,.2f} ({(total_pl/total_value)*100:.2f}%)\n",
            "Current Positions:"
        ]
        
        for pos in portfolio:
            summary.append(
                f"- {pos['stock_symbol']}: {pos['shares']} shares @ ${pos['average_price']:.2f} "
                f"(Current: ${pos['current_price']:.2f}, P/L: ${pos['profit_loss']:.2f})"
            )
        
        return "\n".join(summary)

    def set_current_user(self, account_number: str):
        """Set the current user context."""
        self._current_user = account_number

    def process_request(self, query: str) -> str:
        """Process user requests with enhanced context and security."""
        try:
            # Handle casual conversation first
            query_lower = query.lower().strip()
            
            # Casual conversation patterns
            casual_patterns = {
                r'\b(hi|hello|hey)\b': "Hello! How can I help you with your investments today?",
                r'\b(how are you|how\'s it going)\b': "I'm doing well, thank you! Ready to help you with your financial needs.",
                r'\b(thank you|thanks)\b': "You're welcome! Let me know if you need anything else.",
                r'\b(bye|goodbye)\b': "Goodbye! Have a great day!",
            }
            
            for pattern, response in casual_patterns.items():
                if re.search(pattern, query_lower):
                    return response
            
            # Simple command mapping
            simple_commands = {
                'balance': self._get_balance,
                'portfolio': self.get_portfolio_summary,
                'my balance': self._get_balance,
                'show balance': self._get_balance,
                'check balance': self._get_balance,
                'show portfolio': self.get_portfolio_summary,
                'my portfolio': self.get_portfolio_summary,
                'watchlist': self._get_watchlist,
                'my watchlist': self._get_watchlist,
                'help': self._get_help,
            }
            
            # Check for simple commands
            if query_lower in simple_commands:
                return simple_commands[query_lower]()

            # Get current context
            current_time = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get user data if available
            user_data = {}
            if self._current_user:
                try:
                    portfolio = self.db.get_portfolio(self._current_user)
                    watchlist = self.db.get_watchlist(self._current_user)
                    user = self.db.get_user(self._current_user)
                    user_data = {
                        "account_number": self._current_user,
                        "balance": float(user['balance']),
                        "portfolio": [
                            {
                                "symbol": p['stock_symbol'],
                                "shares": int(p['shares']),
                                "avg_price": float(p['average_price']),
                                "current_price": float(p.get('current_price', 0))
                            } for p in portfolio
                        ],
                        "watchlist": [w['stock_symbol'] for w in watchlist]
                    }
                except Exception as e:
                    print(f"Error getting user data: {e}")
                    user_data = {"error": "Failed to get user data"}
            
            # Handle trade commands
            words = query_lower.split()
            if 'buy' in words or 'sell' in words:
                return self._handle_trade_command(query_lower)
            
            # Handle quote requests
            if query_lower.startswith('quote '):
                symbol = query_lower.split()[1].upper()
                return self._get_stock_quote(symbol)
            
            # Handle watch commands
            if query_lower.startswith('watch '):
                symbol = query_lower.split()[1].upper()
                return self._add_to_watchlist(symbol)
            
            # Process natural language queries
            sentiment = self.analyze_sentiment(query_lower)
            if sentiment["action"] == "CHAT":
                return self._handle_natural_query(query)
            
            # Process through LLM for other queries
            try:
                market_data = self.get_market_data()
                response = self.chain.invoke({
                    "query": query,
                    "current_time": current_time,
                    "user_data": json.dumps(user_data, default=str),
                    "market_data": json.dumps(market_data, default=str),
                    "chat_history": "\n".join(self._chat_history[-5:])
                })
            except Exception as e:
                print(f"LLM error: {e}")
                response = "I'm having trouble processing that request. Please try again or use a specific command."
            
            # Save chat history
            if self._current_user:
                try:
                    self.db.save_chat_message(self._current_user, "USER", query)
                    self.db.save_chat_message(self._current_user, "ASSISTANT", response)
                    
                    self._chat_history.append(f"User: {query}")
                    self._chat_history.append(f"Assistant: {response}")
                except Exception as e:
                    print(f"Error saving chat history: {e}")
            
            return response
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _handle_trade_command(self, query: str) -> str:
        """Handle buy/sell trade commands."""
        try:
            words = query.split()
            action = 'BUY' if 'buy' in words else 'SELL'
            symbol_idx = words.index('buy' if 'buy' in words else 'sell') + 1
            
            # Extract symbol and shares
            shares = None
            symbol = None
            
            for word in words[symbol_idx:]:
                if word.isdigit():
                    shares = int(word)
                elif word.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']:
                    symbol = word.upper()
            
            if not symbol or not shares:
                return "❌ Please specify both the stock symbol and number of shares."
            
            # Get current price with retry logic
            quote = self.db.get_real_time_quote(symbol)
            if quote.get('error'):
                return f"❌ Error: Unable to get quote for {symbol}: {quote['error']}"
            
            # Prepare trade data
            trade_data = {
                "type": "trade",
                "operation": action,
                "data": {
                    "symbol": symbol,
                    "shares": shares,
                    "price": float(quote['price'])
                },
                "natural_response": (
                    f"Would you like to {action.lower()} {shares} shares of {symbol} "
                    f"at ${float(quote['price']):.2f} per share?"
                ),
                "requires_confirmation": True,
                "show_data": True
            }
            
            self._pending_operation = trade_data
            total_cost = float(quote['price']) * shares
            
            return (
                f"{trade_data['natural_response']}\n"
                f"Total {'cost' if action == 'BUY' else 'proceeds'}: ${total_cost:.2f}\n"
                "Please confirm by saying 'yes' or 'confirm'"
            )
            
        except Exception as e:
            return f"❌ Error processing trade: {str(e)}"

    def _get_stock_quote(self, symbol: str) -> str:
        """Get and format stock quote."""
        try:
            quote = self.db.get_real_time_quote(symbol)
            return (
                f"\n📈 {symbol} Quote:\n"
                f"Price: ${float(quote['price']):.2f}\n"
                f"Change: {float(quote['change']):.2f}%\n"
                f"Volume: {int(quote['volume']):,}"
            )
        except Exception as e:
            return f"❌ Error getting quote for {symbol}: {str(e)}"

    def _get_balance(self) -> str:
        """Get user's current balance."""
        if not self._current_user:
            return "Please log in to check your balance."
        try:
            user = self.db.get_user(self._current_user)
            return f"💰 Current Balance: ${float(user['balance']):,.2f}"
        except Exception as e:
            return f"Error getting balance: {str(e)}"

    def _get_watchlist(self) -> str:
        """Get user's watchlist with current prices."""
        if not self._current_user:
            return "Please log in to view your watchlist."
        try:
            watchlist = self.db.get_watchlist(self._current_user)
            if not watchlist:
                return "Your watchlist is empty."
            
            lines = ["📋 Your Watchlist:"]
            for item in watchlist:
                lines.append(
                    f"- {item['stock_symbol']}: ${float(item['price']):.2f} "
                    f"({float(item['change']):.2f}%)"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error getting watchlist: {str(e)}"

    def _add_to_watchlist(self, symbol: str) -> str:
        """Add a stock to user's watchlist."""
        if not self._current_user:
            return "Please log in to modify your watchlist."
        try:
            result = self.db.add_to_watchlist(self._current_user, symbol)
            return f"{'✅' if result['status'] == 'success' else '❌'} {result['message']}"
        except Exception as e:
            return f"Error adding to watchlist: {str(e)}"

    def _get_help(self) -> str:
        """Get help message with available commands."""
        return """
📌 Available Commands:
1. login <email> <password> - Login to account
2. create <email> <password> <account_number> - Create new account
3. portfolio - View your portfolio
4. quote <symbol> - Get stock quote
5. buy <symbol> <shares> - Buy stocks
6. sell <symbol> <shares> - Sell stocks
7. watch <symbol> - Add to watchlist
8. watchlist - View watchlist
9. balance - Check balance
10. deposit <amount> - Deposit funds
11. chat <message> - Chat with agent
12. clear - Clear screen
13. exit - Exit application

You can also ask questions naturally!
"""

    def _execute_operation(self, operation: Dict) -> str:
        """Execute parsed operations with proper error handling."""
        try:
            if not self._current_user:
                return "❌ Please log in to perform this operation."

            if operation["type"] == "trade":
                result = self.db.execute_trade(
                    self._current_user,
                    operation["operation"],
                    operation["data"]["symbol"],
                    operation["data"]["shares"],
                    operation["data"]["price"]
                )
                
                if result["status"] == "success":
                    # Update portfolio summary after trade
                    portfolio = self.db.get_portfolio(self._current_user)
                    return f"✅ {result['message']}\n\n{self._format_portfolio_summary(portfolio)}"
                return f"❌ {result['message']}"
            
            elif operation["type"] == "account":
                if operation["operation"] == "READ":
                    if "portfolio" in operation["data"]:
                        portfolio = self.db.get_portfolio(self._current_user)
                        return self._format_portfolio_summary(portfolio)
                    elif "watchlist" in operation["data"]:
                        watchlist = self.db.get_watchlist(self._current_user)
                        return "Watchlist:\n" + "\n".join(
                            f"- {item['stock_symbol']}: ${item['price']:.2f}" 
                            for item in watchlist
                        )
            
            return operation["natural_response"]
            
        except Exception as e:
            return f"❌ Error executing operation: {str(e)}"

    def confirm_operation(self) -> str:
        """Execute a pending operation after user confirmation."""
        if self._pending_operation:
            operation = self._pending_operation
            self._pending_operation = None
            return self._execute_operation(operation)
        return "No pending operation to confirm."

    def get_portfolio_summary(self) -> str:
        """Get formatted portfolio summary for current user."""
        if not self._current_user:
            return "Please log in to view portfolio."
        
        try:
            portfolio = self.db.get_portfolio(self._current_user)
            if not portfolio:
                return "Your portfolio is empty."
            
            total_value = sum(float(pos['shares']) * float(pos['current_price']) for pos in portfolio)
            total_cost = sum(float(pos['shares']) * float(pos['average_price']) for pos in portfolio)
            total_pl = total_value - total_cost
            
            summary = [
                "📊 Portfolio Summary:",
                f"Total Value: ${total_value:,.2f}",
                f"Total P/L: ${total_pl:,.2f} ({(total_pl/total_cost)*100:.2f}% overall)\n",
                "Current Positions:"
            ]
            
            for pos in portfolio:
                current_value = float(pos['shares']) * float(pos['current_price'])
                cost_basis = float(pos['shares']) * float(pos['average_price'])
                position_pl = current_value - cost_basis
                pl_percent = (position_pl / cost_basis) * 100
                
                summary.append(
                    f"- {pos['stock_symbol']}: {int(pos['shares'])} shares @ ${float(pos['average_price']):.2f} "
                    f"(Current: ${float(pos['current_price']):.2f}, P/L: ${position_pl:.2f} / {pl_percent:.2f}%)"
                )
            
            return "\n".join(summary)
        except Exception as e:
            print(f"Error formatting portfolio: {str(e)}")
            return "Error displaying portfolio."