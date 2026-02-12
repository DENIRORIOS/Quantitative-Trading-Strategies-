//+------------------------------------------------------------------+
//|                                                  TradeManager.mqh |
//|                              Trade Execution and Position Management |
//|                                                                  |
//| Handles:                                                          |
//| - Position sizing based on account equity                        |
//| - Order execution using CTrade class                             |
//| - Long-only strategy implementation                              |
//| - Confidence threshold management                                |
//+------------------------------------------------------------------+

#property copyright "BTC Trading ANN"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Trade Manager Class                                              |
//| Manages all trading operations                                   |
//+------------------------------------------------------------------+
class CTradeManager
{
private:
    CTrade m_trade;
    string m_symbol;
    double m_position_size_pct;
    double m_prediction_threshold;
    int m_magic_number;
    double m_slippage;
    
    // Trade statistics
    int m_total_trades;
    int m_winning_trades;
    int m_losing_trades;
    
public:
    CTradeManager(string symbol, double position_size_pct=0.01, double prediction_threshold=0.65, int magic=12345)
    {
        m_symbol = symbol;
        m_position_size_pct = position_size_pct;
        m_prediction_threshold = prediction_threshold;
        m_magic_number = magic;
        m_slippage = 10;
        
        m_total_trades = 0;
        m_winning_trades = 0;
        m_losing_trades = 0;
        
        // Configure CTrade
        m_trade.SetExpertMagicNumber(m_magic_number);
        m_trade.SetDeviationInPoints((int)m_slippage);
        m_trade.SetTypeFilling(ORDER_FILLING_IOC);
        m_trade.LogLevel(LOG_LEVEL_ERRORS);
    }
    
    //+------------------------------------------------------------------+
    //| Calculate lot size based on account equity                      |
    //+------------------------------------------------------------------+
    double CalculateLotSize()
    {
        double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double position_value = account_equity * m_position_size_pct;
        
        // Get symbol info
        double tick_value = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
        double ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
        double min_lot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
        double max_lot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);
        double lot_step = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
        
        // Calculate lot size
        double lot_size = position_value / (ask * tick_value / tick_size);
        
        // Round to lot step
        lot_size = MathFloor(lot_size / lot_step) * lot_step;
        
        // Clamp to min/max
        lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
        
        return lot_size;
    }
    
    //+------------------------------------------------------------------+
    //| Check if we have an open position                               |
    //+------------------------------------------------------------------+
    bool HasOpenPosition()
    {
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket > 0)
            {
                if(PositionGetString(POSITION_SYMBOL) == m_symbol &&
                   PositionGetInteger(POSITION_MAGIC) == m_magic_number)
                {
                    return true;
                }
            }
        }
        return false;
    }
    
    //+------------------------------------------------------------------+
    //| Get current position ticket                                     |
    //+------------------------------------------------------------------+
    ulong GetPositionTicket()
    {
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket > 0)
            {
                if(PositionGetString(POSITION_SYMBOL) == m_symbol &&
                   PositionGetInteger(POSITION_MAGIC) == m_magic_number)
                {
                    return ticket;
                }
            }
        }
        return 0;
    }
    
    //+------------------------------------------------------------------+
    //| Get current position type                                       |
    //+------------------------------------------------------------------+
    ENUM_POSITION_TYPE GetPositionType()
    {
        ulong ticket = GetPositionTicket();
        if(ticket > 0)
        {
            if(PositionSelectByTicket(ticket))
                return (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        }
        return -1;
    }
    
    //+------------------------------------------------------------------+
    //| Get current position profit                                     |
    //+------------------------------------------------------------------+
    double GetPositionProfit()
    {
        ulong ticket = GetPositionTicket();
        if(ticket > 0)
        {
            if(PositionSelectByTicket(ticket))
                return PositionGetDouble(POSITION_PROFIT);
        }
        return 0.0;
    }
    
    //+------------------------------------------------------------------+
    //| Open long position                                              |
    //+------------------------------------------------------------------+
    bool OpenLong(double prediction_confidence)
    {
        if(HasOpenPosition())
        {
            Print("Cannot open long: position already exists");
            return false;
        }
        
        double lot_size = CalculateLotSize();
        double ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
        
        PrintFormat("Opening LONG position: Lot=%.2f, Price=%.2f, Confidence=%.2f%%",
                    lot_size, ask, prediction_confidence * 100);
        
        bool result = m_trade.Buy(lot_size, m_symbol, ask, 0, 0, 
                                  StringFormat("ANN Long %.2f%%", prediction_confidence * 100));
        
        if(result)
        {
            m_total_trades++;
            Print("Long position opened successfully, ticket: ", m_trade.ResultOrder());
        }
        else
        {
            Print("Failed to open long position, error: ", GetLastError(), 
                  " - ", m_trade.ResultRetcodeDescription());
        }
        
        return result;
    }
    
    //+------------------------------------------------------------------+
    //| Close current position                                          |
    //+------------------------------------------------------------------+
    bool ClosePosition(string reason="")
    {
        ulong ticket = GetPositionTicket();
        if(ticket == 0)
        {
            Print("No position to close");
            return false;
        }
        
        if(!PositionSelectByTicket(ticket))
        {
            Print("Failed to select position");
            return false;
        }
        
        double profit = PositionGetDouble(POSITION_PROFIT);
        double volume = PositionGetDouble(POSITION_VOLUME);
        
        PrintFormat("Closing position: Ticket=%I64u, Profit=%.2f, Reason=%s", 
                    ticket, profit, reason);
        
        bool result = m_trade.PositionClose(ticket);
        
        if(result)
        {
            if(profit > 0)
                m_winning_trades++;
            else if(profit < 0)
                m_losing_trades++;
                
            Print("Position closed successfully");
        }
        else
        {
            Print("Failed to close position, error: ", GetLastError(),
                  " - ", m_trade.ResultRetcodeDescription());
        }
        
        return result;
    }
    
    //+------------------------------------------------------------------+
    //| Execute trading logic based on prediction                       |
    //| Long-only strategy:                                             |
    //| - Open long when P(up) > threshold                              |
    //| - Close when P(down) > threshold or confidence too low          |
    //+------------------------------------------------------------------+
    void ExecuteTrade(double prediction)
    {
        bool has_position = HasOpenPosition();
        
        // Calculate confidence for up and down
        double confidence_up = prediction;
        double confidence_down = 1.0 - prediction;
        
        // Trading logic
        if(!has_position)
        {
            // No position: check if we should open long
            if(confidence_up > m_prediction_threshold)
            {
                OpenLong(confidence_up);
            }
            else
            {
                // No confident signal, stay flat
            }
        }
        else
        {
            // Have position: check if we should close
            ENUM_POSITION_TYPE pos_type = GetPositionType();
            
            if(pos_type == POSITION_TYPE_BUY)
            {
                // Long position: close if bearish signal or low confidence
                if(confidence_down > m_prediction_threshold)
                {
                    ClosePosition(StringFormat("Bearish signal: %.2f%%", confidence_down * 100));
                }
                else if(confidence_up < m_prediction_threshold)
                {
                    ClosePosition(StringFormat("Low confidence: %.2f%%", confidence_up * 100));
                }
            }
        }
    }
    
    //+------------------------------------------------------------------+
    //| Get trade statistics                                            |
    //+------------------------------------------------------------------+
    void PrintStatistics()
    {
        double win_rate = (m_total_trades > 0) ? 
                         (double)m_winning_trades / m_total_trades * 100.0 : 0.0;
        
        PrintFormat("=== Trade Statistics ===");
        PrintFormat("Total Trades: %d", m_total_trades);
        PrintFormat("Winning Trades: %d", m_winning_trades);
        PrintFormat("Losing Trades: %d", m_losing_trades);
        PrintFormat("Win Rate: %.2f%%", win_rate);
        PrintFormat("Account Balance: %.2f", AccountInfoDouble(ACCOUNT_BALANCE));
        PrintFormat("Account Equity: %.2f", AccountInfoDouble(ACCOUNT_EQUITY));
    }
    
    //+------------------------------------------------------------------+
    //| Check if market is open for trading                             |
    //+------------------------------------------------------------------+
    bool IsMarketOpen()
    {
        MqlDateTime dt;
        TimeToStruct(TimeCurrent(), dt);
        
        // Check if symbol is available
        if(!SymbolInfoInteger(m_symbol, SYMBOL_SELECT))
        {
            Print("Symbol not available: ", m_symbol);
            return false;
        }
        
        // Check if trading is allowed
        if(!SymbolInfoInteger(m_symbol, SYMBOL_TRADE_MODE))
        {
            Print("Trading not allowed for symbol: ", m_symbol);
            return false;
        }
        
        // Check session
        datetime from, to;
        if(!SymbolInfoSessionTrade(m_symbol, (ENUM_DAY_OF_WEEK)dt.day_of_week, 0, from, to))
        {
            // If no session info, assume 24/7 (crypto)
            return true;
        }
        
        return true;  // Assume market is open if we got here
    }
    
    //+------------------------------------------------------------------+
    //| Getters                                                         |
    //+------------------------------------------------------------------+
    int GetTotalTrades() { return m_total_trades; }
    int GetWinningTrades() { return m_winning_trades; }
    int GetLosingTrades() { return m_losing_trades; }
    double GetWinRate() { 
        return (m_total_trades > 0) ? (double)m_winning_trades / m_total_trades * 100.0 : 0.0;
    }
};
