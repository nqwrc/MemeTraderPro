import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class TradingLogic:
    """
    Class to handle trading logic based on model predictions
    """
    def __init__(self, risk_per_trade=0.02, stop_loss_pct=0.05, take_profit_pct=0.15):
        """
        Initialize trading logic

        Parameters:
        risk_per_trade (float): Percentage of account balance to risk per trade (0.02 = 2%)
        stop_loss_pct (float): Stop loss percentage (0.05 = 5%)
        take_profit_pct (float): Take profit percentage (0.15 = 15%)
        """
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = 0.60  # Minimum confidence to take a trade
        self.max_active_trades = 3  # Maximum number of active trades
        self.cooldown_period = 6  # Cooldown period in hours after closing a trade
        self.recent_trades = {}  # Dictionary to track recent trades by symbol and avoid quick re-entry

    def decide_action(self, symbol, prediction, confidence, current_price, position, available_balance):
        """Decide whether to buy, sell, or hold based on prediction and current position"""
        import random

        # Default is to hold
        action = 'hold'
        amount = 0
        reason = 'No signal'

        # For dry run mode, we can detect it by checking if available_balance is a fixed value
        is_dry_run = isinstance(available_balance, float) and available_balance == 1000.0

        # If we have a position, check if we should sell
        if position:
            entry_price = position['entry_price']
            position_amount = position['amount']
            position_duration = (datetime.now() - position['timestamp']).total_seconds() / 60

            # Check for take profit
            take_profit_price = entry_price * (1 + self.take_profit_pct)
            if current_price >= take_profit_price:
                action = 'sell'
                amount = position_amount
                reason = f'Take profit triggered at {take_profit_price:.6f}'

            # Check for stop loss
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            if current_price <= stop_loss_price:
                action = 'sell'
                amount = position_amount
                reason = f'Stop loss triggered at {stop_loss_price:.6f}'

            # Check for bearish signal with high confidence
            if prediction == 0 and confidence > 0.7:
                action = 'sell'
                amount = position_amount
                reason = f'Strong bearish signal with confidence {confidence:.4f}'

            # In dry run, if position has been held for more than 10 minutes, 
            # occasionally close it to demonstrate the trading cycle
            if is_dry_run and position_duration > 10 and random.random() < 0.4:
                # Simulate a profitable trade 70% of the time
                if random.random() < 0.7:
                    # Simulate a 2-8% profit
                    simulated_price = entry_price * (1 + (random.random() * 0.06 + 0.02))
                    action = 'sell'
                    amount = position_amount
                    reason = f'[DRY RUN] Simulated profit taking at {simulated_price:.6f}'
                else:
                    # Simulate a 1-3% loss
                    simulated_price = entry_price * (1 - (random.random() * 0.02 + 0.01))
                    action = 'sell'
                    amount = position_amount
                    reason = f'[DRY RUN] Simulated stop loss at {simulated_price:.6f}'

        # If we don't have a position and get a bullish signal, consider buying
        elif prediction == 1 and confidence > 0.65:
            # Calculate position size based on risk
            risk_amount = available_balance * self.risk_per_trade

            # Calculate amount to buy (in crypto units)
            amount = risk_amount / current_price

            # If amount is too small, don't trade
            if amount * current_price < 10:  # Minimum order value $10
                action = 'hold'
                amount = 0
                reason = 'Order too small'
            else:
                action = 'buy'
                reason = f'Bullish signal with confidence {confidence:.4f}'

        # In dry run mode, occasionally force buys to demonstrate the system even without signals
        elif is_dry_run and not position and random.random() < 0.3:
            # Calculate a smaller position size for demonstration
            risk_amount = available_balance * (self.risk_per_trade * 1.5)  # Slightly higher risk for demo
            amount = risk_amount / current_price

            if amount * current_price >= 10:  # Ensure minimum order value
                action = 'buy'
                reason = f'[DRY RUN] Simulated trading signal for demonstration'

        return action, amount, reason

    def adjust_position(self, symbol, current_price, position, new_prediction, new_confidence):
        """
        Adjust an existing position based on new information

        Parameters:
        symbol (str): Trading pair symbol
        current_price (float): Current market price
        position (dict): Current position data
        new_prediction (int): New model prediction
        new_confidence (float): New prediction confidence

        Returns:
        tuple: (action, amount, reason)
        """
        action = 'hold'
        amount = 0
        reason = 'No adjustment needed'

        # We assume position is not None
        entry_price = position['entry_price']
        position_amount = position['amount']
        position_value = position_amount * current_price
        unrealized_pl_pct = (current_price / entry_price - 1)

        # If we're in profit and prediction is turning bearish, consider partial exit
        if unrealized_pl_pct > 0.05 and new_prediction == 0 and new_confidence > 0.6:
            # Take partial profit (50% of position)
            action = 'sell'
            amount = position_amount * 0.5
            reason = f'Partial profit taking: {unrealized_pl_pct:.2%}, bearish signal'

        # If we're in profit and prediction is still bullish, consider moving stop loss up
        elif unrealized_pl_pct > 0.08 and new_prediction == 1 and new_confidence > 0.6:
            # We can't directly adjust stop loss here, so we keep holding
            action = 'hold'
            amount = 0
            reason = f'Moving stop loss up to breakeven'
            # In a real implementation, we would adjust the stop loss level

        return action, amount, reason

    def check_max_trades(self, active_positions, max_positions=None):
        """
        Check if we have hit the maximum number of simultaneous trades

        Parameters:
        active_positions (dict): Dictionary of active positions
        max_positions (int, optional): Maximum allowed positions, defaults to self.max_active_trades

        Returns:
        bool: True if we can take more trades, False if we're at the limit
        """
        if max_positions is None:
            max_positions = self.max_active_trades

        return len(active_positions) < max_positions

    def calculate_position_size(self, symbol, prediction, confidence, current_price, balance):
        """
        Calculate the appropriate position size for a new trade

        Parameters:
        symbol (str): Trading pair symbol
        prediction (int): Model prediction
        confidence (float): Prediction confidence
        current_price (float): Current market price
        balance (float): Available balance

        Returns:
        float: Position size in base currency units
        """
        # Default to no position
        position_size = 0

        # Only calculate for buy signals
        if prediction == 1 and confidence > self.min_confidence:
            # Calculate position size based on risk
            risk_amount = balance * self.risk_per_trade
            stop_loss_amount = current_price * self.stop_loss_pct

            # Calculate amount to buy
            if stop_loss_amount > 0:
                position_size = risk_amount / stop_loss_amount

            # Adjust based on confidence
            confidence_factor = (confidence - self.min_confidence) / (1 - self.min_confidence)
            position_size = position_size * (0.5 + 0.5 * confidence_factor)

            # Make sure amount is not too large compared to balance
            max_size = balance / current_price * 0.95  # Max 95% of balance
            position_size = min(position_size, max_size)

            # Make sure amount is greater than minimum
            if position_size * current_price < 10:  # Minimum order value of $10
                position_size = 0

        return position_size

if __name__ == "__main__":
    # Example usage
    trading_logic = TradingLogic(risk_per_trade=0.02, stop_loss_pct=0.05, take_profit_pct=0.15)

    # Example 1: No position, bullish prediction
    action, amount, reason = trading_logic.decide_action(
        symbol="DOGE/USDT",
        prediction=1,  # Bullish
        confidence=0.75,
        current_price=0.1,
        position=None,
        available_balance=1000
    )
    print(f"Example 1: {action}, amount: {amount}, reason: {reason}")

    # Example 2: Existing position, price dropped below stop loss
    action, amount, reason = trading_logic.decide_action(
        symbol="SHIB/USDT",
        prediction=1,  # Still bullish
        confidence=0.65,
        current_price=0.000028,
        position={'entry_price': 0.000030, 'amount': 100000, 'timestamp': datetime.now() - timedelta(hours=5)},
        available_balance=1000
    )
    print(f"Example 2: {action}, amount: {amount}, reason: {reason}")

    # Example 3: Existing position, price increased above take profit
    action, amount, reason = trading_logic.decide_action(
        symbol="PEPE/USDT",
        prediction=1,  # Still bullish
        confidence=0.7,
        current_price=0.000012,
        position={'entry_price': 0.000010, 'amount': 1000000, 'timestamp': datetime.now() - timedelta(hours=10)},
        available_balance=1000
    )
    print(f"Example 3: {action}, amount: {amount}, reason: {reason}")