import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
    
    def decide_action(self, symbol, prediction, confidence, current_price, position, balance):
        """
        Decide what trading action to take based on model prediction
        
        Parameters:
        symbol (str): Trading pair symbol
        prediction (int): Model prediction (1 for up, 0 for down)
        confidence (float): Prediction confidence (0-1)
        current_price (float): Current market price
        position (dict): Current position data if exists (None otherwise)
        balance (float): Current account balance
        
        Returns:
        tuple: (action, amount, reason)
        """
        # Default response
        action = 'hold'
        amount = 0
        reason = 'No signal'
        
        # Check if we already have a position for this symbol
        if position is not None:
            # We have an existing position, check if we should exit
            entry_price = position['entry_price']
            position_amount = position['amount']
            position_value = position_amount * current_price
            unrealized_pl_pct = (current_price / entry_price - 1)
            
            # Check stop loss
            if unrealized_pl_pct < -self.stop_loss_pct:
                action = 'sell'
                amount = position_amount
                reason = f'Stop loss triggered: {unrealized_pl_pct:.2%}'
            
            # Check take profit
            elif unrealized_pl_pct > self.take_profit_pct:
                action = 'sell'
                amount = position_amount
                reason = f'Take profit triggered: {unrealized_pl_pct:.2%}'
            
            # Check if prediction suggests a reversal with high confidence
            elif prediction == 0 and confidence > 0.75:
                action = 'sell'
                amount = position_amount
                reason = f'Strong sell signal: confidence {confidence:.2f}'
            
            # Otherwise hold position
            else:
                action = 'hold'
                amount = 0
                reason = f'Holding position: current P/L {unrealized_pl_pct:.2%}'
                
        else:
            # We don't have a position, check if we should enter
            
            # Check if symbol is in cooldown period after a recent trade
            if symbol in self.recent_trades:
                last_trade_time = self.recent_trades[symbol]
                if datetime.now() - last_trade_time < timedelta(hours=self.cooldown_period):
                    return 'hold', 0, f'Symbol in cooldown period after recent trade'
            
            # If prediction is 1 (up) and confidence is above threshold
            if prediction == 1 and confidence > self.min_confidence:
                # Calculate position size based on risk
                risk_amount = balance * self.risk_per_trade
                stop_loss_amount = current_price * self.stop_loss_pct
                
                # Calculate amount to buy
                if stop_loss_amount > 0:
                    amount = risk_amount / stop_loss_amount
                else:
                    amount = 0
                
                # Make sure amount is not too large compared to balance
                max_amount = balance / current_price * 0.95  # Max 95% of balance
                amount = min(amount, max_amount)
                
                # Make sure amount is greater than minimum
                if amount * current_price >= 10:  # Minimum order value of $10
                    action = 'buy'
                    reason = f'Buy signal: prediction {prediction}, confidence {confidence:.2f}'
                else:
                    action = 'hold'
                    amount = 0
                    reason = 'Insufficient funds for minimum order'
            else:
                action = 'hold'
                amount = 0
                reason = f'No entry signal: prediction {prediction}, confidence {confidence:.2f}'
        
        # Update recent trades if we're exiting a position
        if action == 'sell':
            self.recent_trades[symbol] = datetime.now()
        
        # Return the decision
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
        balance=1000
    )
    print(f"Example 1: {action}, amount: {amount}, reason: {reason}")
    
    # Example 2: Existing position, price dropped below stop loss
    action, amount, reason = trading_logic.decide_action(
        symbol="SHIB/USDT",
        prediction=1,  # Still bullish
        confidence=0.65,
        current_price=0.000028,
        position={'entry_price': 0.000030, 'amount': 100000, 'timestamp': datetime.now() - timedelta(hours=5)},
        balance=1000
    )
    print(f"Example 2: {action}, amount: {amount}, reason: {reason}")
    
    # Example 3: Existing position, price increased above take profit
    action, amount, reason = trading_logic.decide_action(
        symbol="PEPE/USDT",
        prediction=1,  # Still bullish
        confidence=0.7,
        current_price=0.000012,
        position={'entry_price': 0.000010, 'amount': 1000000, 'timestamp': datetime.now() - timedelta(hours=10)},
        balance=1000
    )
    print(f"Example 3: {action}, amount: {amount}, reason: {reason}")
