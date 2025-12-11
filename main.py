
import pandas as pd
import numpy as np
from hyperliquid.info import Info
from hyperliquid.utils import constants
import datetime
import time
import sys
import threading
import traceback

# --- Configuration ---
SYMBOL = "BTC" 
TIMEFRAME = "5m"
# Dummy capital for simulation logging
INITIAL_CAPITAL = 200.0
LEVERAGE = 40

# --- Follow Line 2in1 Inputs (From backtest_btc.py) ---
BB_PERIOD_1 = 21
BB_DEVIATIONS_1 = 1.00
USE_ATR_FILTER_1 = True
ATR_PERIOD_1 = 5

# --- Type of MA Inputs ---
MA_TYPE = 'EMA'
MA_PERIOD = 21
MA_DEVIATIONS = 1.00
MA_ATR_PERIOD = 5
MA_SOURCE = 'close'

# --- Smart Risk Management Inputs ---
SL_ATR_MULT_UI = 1.2
TP_RR_RATIOS_UI = [1.5, 2.5, 3.5, 5.0]
TP_SIZES_UI = [0.25, 0.25, 0.25, 0.25] 

# --- Flux SR Inputs ---
SR_STRENGTH = 2
SR_ZONE_WIDTH = 2
SR_PIVOT_RANGE = 15

# --- Global Variables ---
last_ws_update = time.time()  # Track last WebSocket message

# --- Indicator Functions (Copied from backtest_btc.py) ---

def calc_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calc_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr = np.maximum(high - low, np.maximum((high - close).abs(), (low - close).abs()))
    return calc_rma(tr, period)

def calc_sma(series, period):
    return series.rolling(window=period).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_stdev(series, period):
    return series.rolling(window=period).std()

def calc_follow_line(df):
    bb_period = BB_PERIOD_1
    bb_dev = BB_DEVIATIONS_1
    use_atr_filter = USE_ATR_FILTER_1
    atr_period = ATR_PERIOD_1
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    sma = calc_sma(close, bb_period)
    std = calc_stdev(close, bb_period)
    bb_upper = sma + (std * bb_dev)
    bb_lower = sma - (std * bb_dev)
    
    atr = calc_atr(df, atr_period)
    
    trend_line = np.zeros(len(df))
    i_trend = np.zeros(len(df))
    buy = np.zeros(len(df))
    sell = np.zeros(len(df))
    
    close_np = close.values
    high_np = high.values
    low_np = low.values
    bb_upper_np = bb_upper.fillna(0).values
    bb_lower_np = bb_lower.fillna(0).values
    atr_np = atr.fillna(0).values
    
    prev_trend_line = 0.0
    prev_i_trend = 0
    first_valid_idx = -1
    
    for i in range(len(df)):
        if not np.isnan(bb_upper_np[i]) and not np.isnan(atr_np[i]) and bb_upper_np[i] != 0:
            first_valid_idx = i
            break
            
    if first_valid_idx == -1: return df

    for i in range(len(df)):
        if i < first_valid_idx: continue
        
        c = close_np[i]
        h = high_np[i]
        l = low_np[i]
        bbu = bb_upper_np[i]
        bbl = bb_lower_np[i]
        atrv = atr_np[i]
        
        if i == first_valid_idx: prev_trend_line = c 
        
        bb_signal = 0
        if c > bbu: bb_signal = 1
        elif c < bbl: bb_signal = -1
        
        curr_trend_line = 0.0
        
        if use_atr_filter:
            if bb_signal == 1:
                curr_trend_line = l - atrv
                if curr_trend_line < prev_trend_line: curr_trend_line = prev_trend_line
            elif bb_signal == -1:
                curr_trend_line = h + atrv
                if curr_trend_line > prev_trend_line: curr_trend_line = prev_trend_line
            else:
                curr_trend_line = prev_trend_line
        else:
            if bb_signal == 1:
                curr_trend_line = l
                if curr_trend_line < prev_trend_line: curr_trend_line = prev_trend_line
            elif bb_signal == -1:
                curr_trend_line = h
                if curr_trend_line > prev_trend_line: curr_trend_line = prev_trend_line
            else:
                curr_trend_line = prev_trend_line
        
        if i == first_valid_idx:
            curr_trend_line = l - atrv if bb_signal == 1 else h + atrv

        trend_line[i] = curr_trend_line
        
        curr_i_trend = prev_i_trend
        if curr_trend_line > prev_trend_line: curr_i_trend = 1
        if curr_trend_line < prev_trend_line: curr_i_trend = -1
        
        prev_trend_line = curr_trend_line
        i_trend[i] = curr_i_trend
        prev_i_trend = curr_i_trend
        
        if i > 0:
            if i_trend[i-1] == -1 and i_trend[i] == 1: buy[i] = 1
            if i_trend[i-1] == 1 and i_trend[i] == -1: sell[i] = 1
            
    df['FollowLine_Buy'] = buy
    df['FollowLine_Sell'] = sell
    df['FollowLine_Trend'] = i_trend
    return df

def calc_type_of_ma(df):
    ma_type = MA_TYPE
    period = MA_PERIOD
    dev = MA_DEVIATIONS
    atr_period = MA_ATR_PERIOD
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    ma = calc_ema(close, period)
        
    std = calc_stdev(close, period)
    bb_upper = ma + (std * dev)
    bb_lower = ma - (std * dev)
    
    atr = calc_atr(df, atr_period)
    
    trend_line = np.zeros(len(df))
    i_trend = np.zeros(len(df))
    buy = np.zeros(len(df))
    sell = np.zeros(len(df))
    
    close_np = close.values
    high_np = high.values
    low_np = low.values
    bb_upper_np = bb_upper.fillna(0).values
    bb_lower_np = bb_lower.fillna(0).values
    atr_np = atr.fillna(0).values
    
    prev_trend_line = 0.0
    prev_i_trend = 0
    first_valid_idx = -1
    
    for i in range(len(df)):
        if not np.isnan(bb_upper_np[i]) and not np.isnan(atr_np[i]) and bb_upper_np[i] != 0:
            first_valid_idx = i
            break
            
    if first_valid_idx == -1: return df
    
    for i in range(len(df)):
        if i < first_valid_idx: continue
        
        c = close_np[i]
        h = high_np[i]
        l = low_np[i]
        bbu = bb_upper_np[i]
        bbl = bb_lower_np[i]
        v = atr_np[i]
        
        if i == first_valid_idx: prev_trend_line = c

        bb_signal = 0
        if c > bbu: bb_signal = 1
        elif c < bbl: bb_signal = -1
        
        curr_trend_line = prev_trend_line
        
        if bb_signal == 1:
            curr_trend_line = l - v
            if curr_trend_line < prev_trend_line: curr_trend_line = prev_trend_line
        elif bb_signal == -1:
            curr_trend_line = h + v
            if curr_trend_line > prev_trend_line: curr_trend_line = prev_trend_line
        
        if i == first_valid_idx:
            curr_trend_line = l - v if bb_signal == 1 else h + v
            
        trend_line[i] = curr_trend_line
        
        curr_i_trend = prev_i_trend
        if curr_trend_line > prev_trend_line: curr_i_trend = 1
        if curr_trend_line < prev_trend_line: curr_i_trend = -1
        
        prev_trend_line = curr_trend_line
        i_trend[i] = curr_i_trend
        prev_i_trend = curr_i_trend
        
        if i > 0:
            if i_trend[i-1] == -1 and i_trend[i] == 1: buy[i] = 1
            if i_trend[i-1] == 1 and i_trend[i] == -1: sell[i] = 1
            
    df['TypeMA_Buy'] = buy
    df['TypeMA_Sell'] = sell
    df['TypeMA_Trend'] = i_trend
    return df

def calc_flux_sr(df):
    pr = SR_PIVOT_RANGE
    high = df['high'].values
    low = df['low'].values
    p_highs = [np.nan] * len(df)
    p_lows = [np.nan] * len(df)
    
    for i in range(pr, len(df) - pr):
        window_high = high[i-pr : i+pr+1]
        window_low = low[i-pr : i+pr+1]
        
        if high[i] == np.max(window_high): p_highs[i] = high[i]
        if low[i] == np.min(window_low): p_lows[i] = low[i]
            
    df['Pivot_High'] = p_highs
    df['Pivot_Low'] = p_lows
    
    df['Roll_Max_300'] = df['high'].rolling(window=300).max()
    df['Roll_Min_300'] = df['low'].rolling(window=300).min()
    df['Zone_Width'] = (df['Roll_Max_300'] - df['Roll_Min_300']) * (SR_ZONE_WIDTH / 100.0)
    
    return df

# --- Trade Class (Reduced to Logging) ---
class Trade:
    def __init__(self, entry_time, entry_price, size, direction, atr, bar_high, bar_low):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.initial_size = size
        self.current_size = size
        self.direction = direction 
        self.active = True
        self.pnl_realized = 0.0
        
        if direction == 1: self.sl_price = bar_low - (atr * SL_ATR_MULT_UI)
        else: self.sl_price = bar_high + (atr * SL_ATR_MULT_UI)
            
        risk_dist = abs(entry_price - self.sl_price)
        self.tps = []
        if direction == 1:
            self.tps = [entry_price + (risk_dist * rr) for rr in TP_RR_RATIOS_UI]
        else:
            self.tps = [entry_price - (risk_dist * rr) for rr in TP_RR_RATIOS_UI]
            
        self.tp_hit_count = 0
        print(f"[{entry_time}] ORDER OPEN: {'BUY' if direction == 1 else 'SELL'} @ {entry_price:.2f} | Size: {size:.4f} | SL: {self.sl_price:.2f} | TPs: {[round(x,2) for x in self.tps]}")

    def update(self, timestamp, high, low, close):
        if not self.active: return
        
        sl_hit = False
        if self.direction == 1:
            if low <= self.sl_price: sl_hit = True
        else:
            if high >= self.sl_price: sl_hit = True
                
        if sl_hit:
            print(f"[{timestamp}] SL HIT @ {self.sl_price:.2f} (Close: {close:.2f})")
            self.active = False
            return

        while self.tp_hit_count < len(self.tps):
            next_tp_idx = self.tp_hit_count
            next_tp_price = self.tps[next_tp_idx]
            
            tp_hit = False
            if self.direction == 1:
                if high >= next_tp_price: tp_hit = True
            else:
                if low <= next_tp_price: tp_hit = True
                
            if tp_hit:
                portion = TP_SIZES_UI[next_tp_idx]
                print(f"[{timestamp}] TP{next_tp_idx+1} HIT @ {next_tp_price:.2f} - Closing {portion*100}%")
                self.tp_hit_count += 1
                if self.tp_hit_count >= len(self.tps):
                    print(f"[{timestamp}] ALL TPs HIT - Search for new trade")
                    self.active = False
            else:
                break 

    def close_signal(self, timestamp, price, reason):
        if not self.active: return
        print(f"[{timestamp}] CLOSE SIGNAL: {reason} @ {price:.2f}")
        self.active = False

# --- Global State ---
df_history = pd.DataFrame()
current_trade = None
active_resistances = []
active_supports = []
MAX_SR = 5
latest_candle_cache = None

def update_strategy(closed_df):
    global current_trade, active_resistances, active_supports
    
    # Run Indicators
    df = closed_df.copy() # Work on copy
    df = calc_follow_line(df)
    df = calc_type_of_ma(df)
    df = calc_flux_sr(df)
    df['ATR'] = calc_atr(df, 14)
    
    curr_bar = df.iloc[-1]
    timestamp = curr_bar.name
    price = curr_bar['close']
    high = curr_bar['high']
    low = curr_bar['low']
    atr = curr_bar['ATR']
    zone_width = curr_bar['Zone_Width']
    
    # SR Update
    if not np.isnan(curr_bar['Pivot_High']):
        active_resistances.append(curr_bar['Pivot_High'])
        if len(active_resistances) > MAX_SR: active_resistances.pop(0)
    if not np.isnan(curr_bar['Pivot_Low']):
        active_supports.append(curr_bar['Pivot_Low'])
        if len(active_supports) > MAX_SR: active_supports.pop(0)
        
    if np.isnan(atr): return
    
    # Trade Update
    if current_trade and current_trade.active:
        current_trade.update(timestamp, high, low, price)
        if not current_trade.active:
            current_trade = None
            
    # Signal Logic
    buy1 = curr_bar['FollowLine_Buy']
    buy_ma = curr_bar['TypeMA_Buy']
    sell1 = curr_bar['FollowLine_Sell']
    sell_ma = curr_bar['TypeMA_Sell']
    
    signal = "HOLD"
    if (buy1 == 1) or (buy_ma == 1): signal = "BUY"
    elif (sell1 == 1) or (sell_ma == 1): signal = "SELL"
    
    if signal != "HOLD":
        is_valid = True
        if signal == "BUY":
            for res in active_resistances:
                if abs(price - res) < zone_width: is_valid = False; break
        elif signal == "SELL":
            for sup in active_supports:
                if abs(price - sup) < zone_width: is_valid = False; break
                
        if is_valid:
            if current_trade and current_trade.active:
                current_trade.close_signal(timestamp, price, "SIGNAL_FLIP")
                current_trade = None
                
            direction = 1 if signal == "BUY" else -1
            size = (INITIAL_CAPITAL * LEVERAGE) / price
            current_trade = Trade(timestamp, price, size, direction, atr, high, low)
            
    print(f"[{timestamp}] P: {price:.2f} | Sig: {signal} | Valid: {is_valid if signal != 'HOLD' else '-'} | SRs: {len(active_resistances)+len(active_supports)}")

def on_candle_update(msg):
    global df_history, latest_candle_cache, last_ws_update
    
    try:
        # Update last message timestamp
        last_ws_update = time.time()
        
        data = msg.get('data', {})
        if not data: 
            print(f"\n[WARNING] Empty data in WebSocket message: {msg}")
            return
        
        # Data format from WS: {'t': 1733837400000, 'T': 1733837699999, 's': 'BTC', 'i': '5m', 'o': 98000.0, 'c': 98100.0, 'h': 98200.0, 'l': 97900.0, 'v': 100.0, 'n': 10}
        # We rely on 't' (start time) to detect new candle
        
        new_t = data['t']
        close_price = float(data['c'])
        open_price = float(data['o'])
        high_price = float(data['h'])
        low_price = float(data['l'])
        volume = float(data['v'])
        
        # If this is the first message or a continuation of the same candle
        if latest_candle_cache is None:
            latest_candle_cache = data
            print(f"\n[INFO] First candle received: {pd.to_datetime(new_t, unit='ms')}")
            return

        last_t = latest_candle_cache['t']
        
        if new_t > last_t:
            # The previous candle (last_t) is now closed.
            # We must finalize it and add it to df_history
            final_candle = latest_candle_cache
            ts = pd.to_datetime(final_candle['t'], unit='ms')
            
            # Adding to DataFrame
            row = pd.DataFrame([{
                'open': float(final_candle['o']),
                'high': float(final_candle['h']),
                'low': float(final_candle['l']),
                'close': float(final_candle['c']),
                'volume': float(final_candle['v'])
            }], index=[ts])
            
            df_history = pd.concat([df_history, row])
            # Ensure we don't grow infinitely, keep last 1000
            if len(df_history) > 1000:
                df_history = df_history.iloc[-1000:]
                
            print(f"\n--- Candle Closed: {ts} | Close: {final_candle['c']} ---")
            update_strategy(df_history)
            
            latest_candle_cache = data
        else:
            # Same candle updating
            latest_candle_cache = data
    except Exception as e:
        print(f"\n[ERROR] in on_candle_update: {e}")
        print(traceback.format_exc())

def main():
    global df_history, last_ws_update
    
    print("Initializing Live BTC Scalper...")
    reconnect_count = 0
    
    while True:  # Reconnection loop
        try:
            if reconnect_count > 0:
                print(f"\n[INFO] Reconnection attempt #{reconnect_count}")
                
            info = Info(constants.MAINNET_API_URL, skip_ws=False)
            
            # 1. Fetch History
            print("Fetching initial history...")
            end_time = int(datetime.datetime.now().timestamp() * 1000)
            start_time = end_time - (1000 * 60 * 5 * 500) # 500 candles
            
            try:
                candles = info.candles_snapshot(name=SYMBOL, interval=TIMEFRAME, startTime=start_time, endTime=end_time)
            except Exception as fetch_error:
                error_msg = str(fetch_error).lower()
                if 'rate limit' in error_msg or '429' in error_msg:
                    print(f"\n[ERROR] RATE LIMIT HIT: {fetch_error}")
                    print("Waiting 60 seconds before retrying...")
                    time.sleep(60)
                    reconnect_count += 1
                    continue
                elif 'connection' in error_msg or 'timeout' in error_msg or 'network' in error_msg:
                    print(f"\n[ERROR] CONNECTION FAILED: {fetch_error}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    reconnect_count += 1
                    continue
                else:
                    print(f"\n[ERROR] Failed to fetch candles: {fetch_error}")
                    raise
            
            df = pd.DataFrame(candles)
            if not df.empty:
                for col in ['o', 'h', 'l', 'c', 'v']:
                    df[col] = pd.to_numeric(df[col])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                df_history = df
                print(f"✓ Loaded {len(df)} candles. Latest: {df.index[-1]}")
                # Run initial strategy to populate supports/resistances
                update_strategy(df_history)
            else:
                print("[WARNING] No history found, continuing anyway...")
            
            # 2. Subscribe
            print(f"Subscribing to {SYMBOL} {TIMEFRAME} candles...")
            subscription = {"type": "candle", "coin": SYMBOL, "interval": TIMEFRAME}
            
            try:
                info.subscribe(subscription, on_candle_update)
                print("✓ WebSocket subscription successful")
                reconnect_count = 0  # Reset on successful connection
            except Exception as sub_error:
                error_msg = str(sub_error).lower()
                if 'rate limit' in error_msg or '429' in error_msg:
                    print(f"\n[ERROR] RATE LIMIT on WebSocket subscription: {sub_error}")
                    print("Waiting 60 seconds before retrying...")
                    time.sleep(60)
                    reconnect_count += 1
                    continue
                elif 'connection' in error_msg or 'refused' in error_msg:
                    print(f"\n[ERROR] WebSocket CONNECTION REFUSED: {sub_error}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    reconnect_count += 1
                    continue
                else:
                    print(f"\n[ERROR] WebSocket subscription failed: {sub_error}")
                    raise
            
            # Initialize last update time
            last_ws_update = time.time()
            
            print("Listening for updates... (Press Ctrl+C to stop)")
            last_heartbeat = time.time()
            ws_timeout_seconds = 90  # If no WS update in 90s, reconnect
            
            while True:
                try:
                    time.sleep(1)
                    
                    # Check for WebSocket timeout
                    time_since_update = time.time() - last_ws_update
                    if time_since_update > ws_timeout_seconds:
                        print(f"\n[ERROR] WebSocket TIMEOUT - No updates for {time_since_update:.0f}s")
                        print("[INFO] Initiating reconnection...")
                        reconnect_count += 1
                        break  # Break inner loop to trigger reconnection
                    
                    # Optional: Print ticker price occasionally
                    if latest_candle_cache:
                        t_ms = latest_candle_cache['t']
                        dt_obj = datetime.datetime.fromtimestamp(t_ms/1000)
                        next_close = dt_obj + datetime.timedelta(minutes=5)
                        time_str = datetime.datetime.now().strftime('%H:%M:%S')
                        
                        # Ticker (Updates same line)
                        sys.stdout.write(f"\r[{time_str}] Price: {latest_candle_cache['c']} | Next close: {next_close.strftime('%H:%M')} | WS: {time_since_update:.0f}s ago     ")
                        sys.stdout.flush()
                        
                        # Heartbeat (New line every 60s)
                        if time.time() - last_heartbeat > 60:
                            sys.stdout.write(f"\n[{time_str}] ✓ STATUS OK | Price: {latest_candle_cache['c']} | Active Trade: {'YES' if current_trade and current_trade.active else 'NO'}\n")
                            last_heartbeat = time.time()
                except KeyboardInterrupt:
                    print("\n[INFO] Shutdown requested by user")
                    return  # Exit completely
                    
        except KeyboardInterrupt:
            print("\n[INFO] Shutdown requested by user")
            break  # Exit reconnection loop
        except ConnectionError as conn_err:
            print(f"\n[ERROR] CONNECTION ERROR: {conn_err}")
            print(f"Reconnecting in 10 seconds... (Attempt #{reconnect_count + 1})")
            time.sleep(10)
            reconnect_count += 1
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or '429' in error_msg:
                print(f"\n[ERROR] RATE LIMIT EXCEEDED: {e}")
                print("Waiting 60 seconds before retrying...")
                time.sleep(60)
            elif 'permission' in error_msg or 'forbidden' in error_msg or '403' in error_msg:
                print(f"\n[ERROR] PERMISSION DENIED: {e}")
                print("Check API access and keys. Waiting 30 seconds...")
                time.sleep(30)
            elif 'timeout' in error_msg or 'timed out' in error_msg:
                print(f"\n[ERROR] REQUEST TIMEOUT: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"\n[ERROR] Unexpected error: {e}")
                print(traceback.format_exc())
                print("Reconnecting in 10 seconds...")
                time.sleep(10)
            reconnect_count += 1

if __name__ == "__main__":
    main()
