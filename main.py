import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class ComprehensiveOptionsAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.05

    def analyze_option(self, ticker, strike_price, days_to_expiry, option_type='call', investment_amount=1000):
        """Analyze an option with comprehensive metrics"""
        print(f"\nAnalyzing {ticker} {option_type} option...")

        # Get stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='3mo')
        current_price = hist['Close'].iloc[-1]

        # Calculate price movement statistics
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        avg_daily_move = abs(daily_returns).mean()
        max_daily_gain = daily_returns.max()
        max_daily_loss = daily_returns.min()

        # Calculate probability metrics
        time_to_expiry = days_to_expiry / 365.0
        std_dev = volatility * np.sqrt(time_to_expiry)
        if option_type == 'call':
            prob_profit = 1 - norm.cdf(np.log(strike_price / current_price) / std_dev)
        else:
            prob_profit = norm.cdf(np.log(strike_price / current_price) / std_dev)

        # Create analysis results
        analysis = {
            'ticker': ticker,
            'current_price': current_price,
            'strike_price': strike_price,
            'days_to_expiry': days_to_expiry,
            'volatility': volatility,
            'avg_daily_move': avg_daily_move,
            'max_daily_gain': max_daily_gain,
            'max_daily_loss': max_daily_loss,
            'prob_profit': prob_profit,
            'investment_amount': investment_amount,
            'option_type': option_type
        }

        # Plot analysis
        self.plot_comprehensive_analysis(analysis, hist)

        return analysis

    def plot_comprehensive_analysis(self, analysis, price_history):
        """Create comprehensive visualizations"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Price History and Zones
        ax1.plot(price_history.index, price_history['Close'], 'b-', label='Stock Price')
        ax1.axhline(y=analysis['strike_price'], color='r', linestyle='--',
                    label=f'Strike Price (${analysis["strike_price"]:.2f})')
        ax1.plot(price_history.index[-1], analysis['current_price'], 'go', markersize=10,
                 label=f'Current Price (${analysis["current_price"]:.2f})')

        # Add profit/loss zones
        if analysis['option_type'] == 'call':
            ax1.fill_between(price_history.index, analysis['strike_price'], ax1.get_ylim()[1],
                             alpha=0.1, color='g', label='Profit Zone')
        else:
            ax1.fill_between(price_history.index, ax1.get_ylim()[0], analysis['strike_price'],
                             alpha=0.1, color='g', label='Profit Zone')

        ax1.set_title('Stock Price History and Profit Zones', fontsize=12)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Profit/Loss Scenarios
        price_range = np.linspace(analysis['current_price'] * 0.7,
                                  analysis['current_price'] * 1.3, 100)
        if analysis['option_type'] == 'call':
            profit = np.maximum(price_range - analysis['strike_price'], 0) - \
                     (analysis['current_price'] - analysis['strike_price'])
        else:
            profit = np.maximum(analysis['strike_price'] - price_range, 0) - \
                     (analysis['strike_price'] - analysis['current_price'])

        ax2.plot(price_range, profit, 'b-', label='Profit/Loss')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=analysis['current_price'], color='g', linestyle='--',
                    label='Current Price')
        ax2.axvline(x=analysis['strike_price'], color='r', linestyle='--',
                    label='Strike Price')

        ax2.set_title('Potential Profit/Loss Scenarios', fontsize=12)
        ax2.set_xlabel('Stock Price at Expiration ($)')
        ax2.set_ylabel('Profit/Loss ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()


def explain_analysis(analysis):
    """Provide comprehensive explanation in plain English"""
    explanation = "\n=== COMPREHENSIVE OPTION ANALYSIS ===\n"

    # 1. Basic Information
    explanation += "\n🔍 BASIC INFORMATION:\n"
    explanation += f"Stock: {analysis['ticker']}\n"
    explanation += f"Current Stock Price: ${analysis['current_price']:.2f}\n"
    explanation += f"Strike Price: ${analysis['strike_price']:.2f}\n"
    explanation += f"Days until Expiration: {analysis['days_to_expiry']}\n"
    explanation += f"Option Type: {analysis['option_type'].upper()}\n"

    # 2. Position Status
    explanation += "\n📊 POSITION STATUS:\n"
    if analysis['option_type'] == 'call':
        if analysis['current_price'] > analysis['strike_price']:
            explanation += "✅ IN THE MONEY: The stock price is above your strike price\n"
            explanation += f"   You're currently up ${analysis['current_price'] - analysis['strike_price']:.2f} per share\n"
        else:
            explanation += "⚠️ OUT OF THE MONEY: The stock price is below your strike price\n"
            explanation += f"   You need the stock to rise ${analysis['strike_price'] - analysis['current_price']:.2f} to break even\n"
    else:
        if analysis['current_price'] < analysis['strike_price']:
            explanation += "✅ IN THE MONEY: The stock price is below your strike price\n"
            explanation += f"   You're currently up ${analysis['strike_price'] - analysis['current_price']:.2f} per share\n"
        else:
            explanation += "⚠️ OUT OF THE MONEY: The stock price is above your strike price\n"
            explanation += f"   You need the stock to fall ${analysis['current_price'] - analysis['strike_price']:.2f} to break even\n"

    # 3. Risk Assessment
    explanation += "\n⚠️ RISK ASSESSMENT:\n"
    explanation += f"Market Volatility: {analysis['volatility']:.1%}\n"

    risk_level = "LOW" if analysis['volatility'] < 0.15 else "MEDIUM" if analysis['volatility'] < 0.3 else "HIGH"
    explanation += f"Risk Level: {risk_level}\n"

    explanation += "\nDaily Price Movements:\n"
    explanation += f"- Average: {analysis['avg_daily_move']:.1%}\n"
    explanation += f"- Largest Gain: {analysis['max_daily_gain']:.1%}\n"
    explanation += f"- Largest Loss: {analysis['max_daily_loss']:.1%}\n"

    # 4. Profit Potential
    explanation += "\n💰 PROFIT POTENTIAL:\n"
    explanation += f"Probability of Profit: {analysis['prob_profit']:.1%}\n"

    # 5. Scenarios
    explanation += "\n🎯 POSSIBLE SCENARIOS:\n"
    price_up_10 = analysis['current_price'] * 1.10
    price_down_10 = analysis['current_price'] * 0.90

    explanation += "If the stock goes up 10%:\n"
    if analysis['option_type'] == 'call':
        profit = max(price_up_10 - analysis['strike_price'], 0)
        explanation += f"- Option would be worth: ${profit:.2f} per share\n"
    else:
        profit = max(analysis['strike_price'] - price_up_10, 0)
        explanation += f"- Option would be worth: ${profit:.2f} per share\n"

    explanation += "\nIf the stock goes down 10%:\n"
    if analysis['option_type'] == 'call':
        profit = max(price_down_10 - analysis['strike_price'], 0)
        explanation += f"- Option would be worth: ${profit:.2f} per share\n"
    else:
        profit = max(analysis['strike_price'] - price_down_10, 0)
        explanation += f"- Option would be worth: ${profit:.2f} per share\n"

    # 6. Recommendations
    explanation += "\n💡 RECOMMENDATIONS:\n"
    if analysis['days_to_expiry'] < 7:
        explanation += "⚡ IMMEDIATE ATTENTION NEEDED: Very close to expiration\n"
    elif analysis['days_to_expiry'] < 30:
        explanation += "👀 MONITOR CLOSELY: Less than 30 days to expiration\n"
    else:
        explanation += "✅ TIME IS ON YOUR SIDE: More than 30 days to expiration\n"

    if risk_level == "HIGH":
        explanation += "- Consider using stop losses to manage risk\n"
        explanation += "- Think about taking partial profits if available\n"
    elif risk_level == "MEDIUM":
        explanation += "- Monitor the position regularly\n"
        explanation += "- Have a clear exit strategy\n"
    else:
        explanation += "- Stay within your investment plan\n"
        explanation += "- Watch for changes in market conditions\n"

    return explanation


def main():
    analyzer = ComprehensiveOptionsAnalyzer()

    analysis = analyzer.analyze_option(
        ticker='AAPL',
        strike_price=170,
        days_to_expiry=30,
        option_type='call',
        investment_amount=1000
    )

    print(explain_analysis(analysis))


if __name__ == "__main__":
    main()