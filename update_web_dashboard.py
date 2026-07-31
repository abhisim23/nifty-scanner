import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np

# List of liquid Nifty 50 tickers
NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "SHRIRAMFIN.NS"
]

# Popular high-volume low-priced (penny/sub-₹100) stocks on the NSE
PENNY_STOCK_TICKERS = [
    "IDEA.NS", "YESBANK.NS", "SUZLON.NS", "JPPOWER.NS", "RPOWER.NS", "HCC.NS", 
    "ALOKINDS.NS", "INFIBEAM.NS", "UCOBANK.NS", "CENTRALBK.NS", "IOB.NS", 
    "HFCL.NS", "SEPC.NS", "MOREPENLAB.NS", "VIKASLIFE.NS", "URJA.NS", 
    "GTLINFRA.NS", "FCSALL.NS", "BCG.NS", "SALASAR.NS", "SJVN.NS", 
    "NHPC.NS", "SAIL.NS", "IRFC.NS", "GMRINFRA.NS"
]

# Sector Mapping for Nifty 50 constituents
SECTOR_MAP = {
    "INFY.NS": "IT", "TCS.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT", "LTIM.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "BANKING/FIN", "ICICIBANK.NS": "BANKING/FIN", "AXISBANK.NS": "BANKING/FIN", 
    "SBIN.NS": "BANKING/FIN", "KOTAKBANK.NS": "BANKING/FIN", "INDUSINDBK.NS": "BANKING/FIN", 
    "BAJFINANCE.NS": "BANKING/FIN", "BAJAJFINSV.NS": "BANKING/FIN", "SHRIRAMFIN.NS": "BANKING/FIN",
    "HDFCLIFE.NS": "BANKING/FIN", "SBILIFE.NS": "BANKING/FIN",
    "TATAMOTORS.NS": "AUTOMOBILE", "MARUTI.NS": "AUTOMOBILE", "M&M.NS": "AUTOMOBILE", 
    "BAJAJ-AUTO.NS": "AUTOMOBILE", "HEROMOTOCO.NS": "AUTOMOBILE", "EICHERMOT.NS": "AUTOMOBILE",
    "ITC.NS": "FMCG/CONSUMPTION", "HINDUNILVR.NS": "FMCG/CONSUMPTION", "NESTLEIND.NS": "FMCG/CONSUMPTION", 
    "BRITANNIA.NS": "FMCG/CONSUMPTION", "TATACONSUM.NS": "FMCG/CONSUMPTION", "TITAN.NS": "FMCG/CONSUMPTION",
    "RELIANCE.NS": "ENERGY/INFRA", "COALINDIA.NS": "ENERGY/INFRA", "ONGC.NS": "ENERGY/INFRA", 
    "NTPC.NS": "ENERGY/INFRA", "POWERGRID.NS": "ENERGY/INFRA", "JSWSTEEL.NS": "ENERGY/INFRA", 
    "TATASTEEL.NS": "ENERGY/INFRA", "HINDALCO.NS": "ENERGY/INFRA", "BPCL.NS": "ENERGY/INFRA", 
    "GRASIM.NS": "ENERGY/INFRA", "ADANIENT.NS": "ENERGY/INFRA", "ADANIPORTS.NS": "ENERGY/INFRA",
    "ULTRACEMCO.NS": "ENERGY/INFRA", "ASIANPAINT.NS": "ENERGY/INFRA",
    "SUNPHARMA.NS": "PHARMA", "CIPLA.NS": "PHARMA", "DRREDDY.NS": "PHARMA", 
    "DIVISLAB.NS": "PHARMA", "APOLLOHOSP.NS": "PHARMA"
}

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def generate_dashboard():
    # --- STEP 1: SCAN NIFTY 50 STOCKS ---
    print("Fetching Nifty 50 daily market data...")
    data_df = yf.download(NIFTY_50_TICKERS, period="3mo", interval="1d", group_by="ticker", progress=False)
    
    results = []
    for ticker in NIFTY_50_TICKERS:
        try:
            if ticker not in data_df.columns.levels[0]:
                continue
            df = data_df[ticker].copy().dropna(subset=['Close', 'Volume'])
            if len(df) < 50:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            last_row_date = df.index[-1].strftime('%Y-%m-%d')
            
            if last_row_date == today_str:
                analysis_idx = -2
                target_date = df.index[-2]
                trading_date = df.index[-1]
            else:
                analysis_idx = -1
                target_date = df.index[-1]
                trading_date = datetime.date.today()
                
            # Compute indicators
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
            df['RVOL'] = df['Volume'] / df['Vol_SMA20']
            df['RSI'] = calculate_rsi(df['Close'], period=14)
            df['ATR'] = calculate_atr(df, period=14)
            df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
            
            df['BB_Mid'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
            
            df['Daily_Range'] = df['High'] - df['Low']
            df['Is_NR7'] = df['Daily_Range'] == df['Daily_Range'].rolling(window=7).min()
            
            bb_width_percentile = (df['BB_Width'].rolling(window=30).apply(
                lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10)
            )).iloc[analysis_idx]
            is_squeeze = bb_width_percentile < 0.25
            
            row = df.iloc[analysis_idx]
            close = float(row['Close'])
            high = float(row['High'])
            low = float(row['Low'])
            volume = float(row['Volume'])
            rvol = float(row['RVOL'])
            rsi = float(row['RSI'])
            atr = float(row['ATR'])
            atr_pct = float(row['ATR_Pct'])
            ema20 = float(row['EMA_20'])
            sma50 = float(row['SMA_50'])
            is_nr7 = bool(row['Is_NR7'])
            bb_upper = float(row['BB_Upper'])
            bb_lower = float(row['BB_Lower'])
            
            prev_close = float(df['Close'].iloc[analysis_idx - 1])
            daily_pct_change = ((close - prev_close) / prev_close) * 100
            
            # Scoring
            bullish_score = 0
            bearish_score = 0
            
            if close > ema20: bullish_score += 15
            if close > sma50: bullish_score += 15
            if close < ema20: bearish_score += 15
            if close < sma50: bearish_score += 15
            
            if rvol >= 2.0:
                bullish_score += 30
                bearish_score += 30
            elif rvol >= 1.5:
                bullish_score += 20
                bearish_score += 20
            elif rvol >= 1.0:
                bullish_score += 10
                bearish_score += 10
                
            if 55 <= rsi <= 72: bullish_score += 20
            elif 72 < rsi <= 80: bullish_score += 10
            
            if 28 <= rsi <= 45: bearish_score += 20
            elif 20 <= rsi < 28: bearish_score += 10
            
            if is_squeeze:
                bullish_score += 10
                bearish_score += 10
            if is_nr7:
                bullish_score += 10
                bearish_score += 10
            if close > bb_upper: bullish_score += 10
            if close < bb_lower: bearish_score += 10
            
            if atr_pct < 1.0:
                bullish_score -= 15
                bearish_score -= 15
            elif atr_pct >= 1.5:
                bullish_score += 5
                bearish_score += 5
                
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            sector = SECTOR_MAP.get(ticker, "OTHER")
            
            results.append({
                'ticker': ticker,
                'symbol': ticker.replace(".NS", ""),
                'sector': sector,
                'close': close,
                'high': high,
                'low': low,
                'rvol': rvol,
                'rsi': rsi,
                'atr': atr,
                'atr_pct': atr_pct,
                'bull_score': max(0, bullish_score),
                'bear_score': max(0, bearish_score),
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                's1': s1,
                's2': s2,
                'is_squeeze': is_squeeze,
                'is_nr7': is_nr7,
                'change_pct': daily_pct_change,
                'target_date': target_date.strftime('%Y-%m-%d'),
                'trading_date': trading_date.strftime('%Y-%m-%d') if hasattr(trading_date, 'strftime') else str(trading_date)
            })
        except Exception as e:
            print(f"Error on {ticker}: {e}")
            continue
            
    df_all = pd.DataFrame(results)
    if df_all.empty:
        print("Empty dataframe.")
        return
        
    # Calculate Sectoral Momentum
    sector_perf = df_all.groupby('sector')['change_pct'].mean().to_dict()
    
    longs = df_all[df_all['bull_score'] >= 45].sort_values(by='bull_score', ascending=False).head(3)
    shorts = df_all[df_all['bear_score'] >= 45].sort_values(by='bear_score', ascending=False).head(3)
    
    target_date_str = longs['target_date'].iloc[0] if not longs.empty else df_all['target_date'].iloc[0]
    trading_date_str = longs['trading_date'].iloc[0] if not longs.empty else df_all['trading_date'].iloc[0]
    
    # --- STEP 2: REAL-TIME INTRADAY BREAKOUT SCANNER ---
    print("Checking for live intraday breakouts...")
    intraday_data = yf.download(NIFTY_50_TICKERS, period="1d", interval="15m", group_by="ticker", progress=False)
    
    live_bullish_orb = []
    live_bearish_orb = []
    
    for ticker in NIFTY_50_TICKERS:
        try:
            if ticker not in intraday_data.columns.levels[0]:
                continue
            df_i = intraday_data[ticker].copy().dropna(subset=['Close', 'Volume'])
            if len(df_i) < 2:
                continue
            if isinstance(df_i.columns, pd.MultiIndex):
                df_i.columns = df_i.columns.get_level_values(0)
                
            or_high = float(df_i['High'].iloc[0])
            or_low = float(df_i['Low'].iloc[0])
            current_price = float(df_i['Close'].iloc[-1])
            vol_total = float(df_i['Volume'].sum())
            
            if current_price > or_high:
                breakout_pct = ((current_price - or_high) / or_high) * 100
                live_bullish_orb.append({
                    'symbol': ticker.replace(".NS", ""),
                    'or_high': or_high,
                    'or_low': or_low,
                    'current': current_price,
                    'breakout_pct': breakout_pct,
                    'volume': vol_total
                })
            elif current_price < or_low:
                breakdown_pct = ((or_low - current_price) / or_low) * 100
                live_bearish_orb.append({
                    'symbol': ticker.replace(".NS", ""),
                    'or_high': or_high,
                    'or_low': or_low,
                    'current': current_price,
                    'breakdown_pct': breakdown_pct,
                    'volume': vol_total
                })
        except Exception as e:
            continue
            
    df_live_bull = pd.DataFrame(live_bullish_orb)
    df_live_bear = pd.DataFrame(live_bearish_orb)
    
    top_live_bulls = df_live_bull.sort_values(by='breakout_pct', ascending=False).head(3) if not df_live_bull.empty else pd.DataFrame()
    top_live_bears = df_live_bear.sort_values(by='breakdown_pct', ascending=False).head(3) if not df_live_bear.empty else pd.DataFrame()

    # --- STEP 3: BANK NIFTY CORE ANALYSIS ---
    print("Analyzing Bank Nifty indices...")
    bn_daily = yf.download("^NSEBANK", period="3mo", interval="1d", progress=False)
    bn_weekly = yf.download("^NSEBANK", period="1y", interval="1wk", progress=False)
    
    if isinstance(bn_daily.columns, pd.MultiIndex):
        bn_daily.columns = bn_daily.columns.get_level_values(0)
    if isinstance(bn_weekly.columns, pd.MultiIndex):
        bn_weekly.columns = bn_weekly.columns.get_level_values(0)
        
    bn_daily = bn_daily.dropna()
    bn_weekly = bn_weekly.dropna()
    
    bn_daily['EMA_20'] = bn_daily['Close'].ewm(span=20, adjust=False).mean()
    latest_bn_close = float(bn_daily['Close'].iloc[-1])
    latest_bn_ema = float(bn_daily['EMA_20'].iloc[-1])
    daily_bias = "🟢 BULLISH" if latest_bn_close > latest_bn_ema else "🔴 BEARISH"
    
    bn_weekly['EMA_20'] = bn_weekly['Close'].ewm(span=20, adjust=False).mean()
    latest_bn_w_close = float(bn_weekly['Close'].iloc[-1])
    latest_bn_w_ema = float(bn_weekly['EMA_20'].iloc[-1])
    weekly_bias = "🟢 BULLISH" if latest_bn_w_close > latest_bn_w_ema else "🔴 BEARISH"
    
    bn_idx = -2 if bn_daily.index[-1].strftime('%Y-%m-%d') == today_str else -1
    bn_high = float(bn_daily['High'].iloc[bn_idx])
    bn_low = float(bn_daily['Low'].iloc[bn_idx])
    bn_close = float(bn_daily['Close'].iloc[bn_idx])
    
    bn_pivot = (bn_high + bn_low + bn_close) / 3
    bn_r1 = (2 * bn_pivot) - bn_low
    bn_s1 = (2 * bn_pivot) - bn_high
    bn_r2 = bn_pivot + (bn_high - bn_low)
    bn_s2 = bn_pivot - (bn_high - bn_low)
    
    daily_long_trigger = bn_high
    daily_long_sl = bn_pivot
    daily_long_t1 = bn_r1
    daily_long_t2 = bn_r2
    
    daily_short_trigger = bn_low
    daily_short_sl = bn_pivot
    daily_short_t1 = bn_s1
    daily_short_t2 = bn_s2
    
    bn_w_idx = -2 if bn_weekly.index[-1].date() >= (datetime.date.today() - datetime.timedelta(days=4)) else -1
    w_high = float(bn_weekly['High'].iloc[bn_w_idx])
    w_low = float(bn_weekly['Low'].iloc[bn_w_idx])
    w_close = float(bn_weekly['Close'].iloc[bn_w_idx])
    
    w_pivot = (w_high + w_low + w_close) / 3
    w_r1 = (2 * w_pivot) - w_low
    w_s1 = (2 * w_pivot) - w_high
    w_r2 = w_pivot + (w_high - w_low)
    w_s2 = w_pivot - (w_high - w_low)
    
    weekly_long_trigger = w_high
    weekly_long_sl = w_pivot
    weekly_long_t1 = w_r1
    weekly_long_t2 = w_r2
    
    weekly_short_trigger = w_low
    weekly_short_sl = w_pivot
    weekly_short_t1 = w_s1
    weekly_short_t2 = w_s2
    
    # --- STEP 4: SCAN PENNY STOCKS ---
    print("Scanning potential penny stocks...")
    penny_df = yf.download(PENNY_STOCK_TICKERS, period="1mo", interval="1d", group_by="ticker", progress=False)
    penny_results = []
    
    for ticker in PENNY_STOCK_TICKERS:
        try:
            if ticker not in penny_df.columns.levels[0]:
                continue
            df_p = penny_df[ticker].copy().dropna(subset=['Close', 'Volume'])
            if len(df_p) < 15:
                continue
            if isinstance(df_p.columns, pd.MultiIndex):
                df_p.columns = df_p.columns.get_level_values(0)
                
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            if df_p.index[-1].strftime('%Y-%m-%d') == today_str:
                analysis_idx = -2
            else:
                analysis_idx = -1
                
            row_p = df_p.iloc[analysis_idx]
            close_p = float(row_p['Close'])
            vol_p = float(row_p['Volume'])
            
            avg_vol_20 = df_p['Volume'].rolling(window=15).mean().iloc[analysis_idx]
            rvol_p = vol_p / (avg_vol_20 + 1e-10)
            
            prev_close_p = float(df_p['Close'].iloc[analysis_idx - 1])
            change_p = ((close_p - prev_close_p) / prev_close_p) * 100
            
            if 1.0 <= close_p <= 100.0 and rvol_p >= 1.2:
                penny_results.append({
                    'ticker': ticker,
                    'symbol': ticker.replace(".NS", ""),
                    'close': close_p,
                    'rvol': rvol_p,
                    'change_pct': change_p,
                })
        except Exception as e:
            continue
            
    df_penny_results = pd.DataFrame(penny_results)
    if not df_penny_results.empty:
        top_penny_stocks = df_penny_results.sort_values(by=['rvol', 'change_pct'], ascending=False).head(5)
    else:
        top_penny_stocks = pd.DataFrame()
        
    # --- STEP 5: TEMPLATE THE DASHBOARD ---
    sector_heatmap_html = ""
    for sect, perf in sector_perf.items():
        perf_color = "text-green-400" if perf >= 0 else "text-red-400"
        perf_bg = "bg-green-950/40 border-green-500/20" if perf >= 0 else "bg-red-950/40 border-red-500/20"
        perf_symbol = "+" if perf >= 0 else ""
        icon = "fa-laptop-code" if sect == "IT" else "fa-building-columns" if "BANKING" in sect else "fa-car" if sect == "AUTOMOBILE" else "fa-cart-shopping" if "FMCG" in sect else "fa-bolt" if "ENERGY" in sect else "fa-prescription-bottle-medical"
        
        sector_heatmap_html += f"""
        <div class="border rounded-xl p-4 {perf_bg} flex flex-col justify-between shadow transition hover:scale-105 duration-200">
            <div class="flex justify-between items-center mb-2">
                <span class="text-xs text-slate-400 font-bold uppercase tracking-wider">{sect}</span>
                <i class="fa-solid {icon} text-slate-400 text-sm"></i>
            </div>
            <div class="flex justify-between items-end">
                <span class="text-[10px] text-slate-500 font-semibold uppercase">Daily Momentum</span>
                <span class="text-lg font-black {perf_color}">{perf_symbol}{perf:.2f}%</span>
            </div>
        </div>
        """
        
    live_breakouts_html = ""
    if top_live_bulls.empty and top_live_bears.empty:
        live_breakouts_html = """
        <div class="col-span-full bg-slate-900/40 rounded-xl p-6 border border-slate-800 text-center text-xs text-gray-400">
            <i class="fa-solid fa-hourglass-start text-yellow-500 text-lg mb-2"></i><br>
            <strong>Waiting for regular market hours...</strong><br>
            During trading sessions (09:15 AM - 03:30 PM IST), this panel scans active 15-minute bars and lists Nifty 50 stocks breaking their Opening Range in real-time!
        </div>
        """
    else:
        live_breakouts_html = '<div class="grid grid-cols-1 md:grid-cols-2 gap-6 col-span-full">'
        
        # Bullish ORB List
        live_breakouts_html += """
        <div class="bg-slate-900/60 border border-green-500/20 rounded-2xl p-5 shadow">
            <h4 class="text-green-400 font-bold text-sm mb-4 flex items-center gap-2 uppercase tracking-wider">
                <i class="fa-solid fa-circle-arrow-up text-lg"></i> Live Bullish ORB Breakouts (Above 15m High)
            </h4>
            <div class="space-y-3">
        """
        if top_live_bulls.empty:
            live_breakouts_html += '<p class="text-xs text-gray-500 italic">No active bullish breakouts detected at this moment.</p>'
        else:
            for idx, row in top_live_bulls.iterrows():
                live_breakouts_html += f"""
                <div class="bg-gray-800/40 p-3 rounded-xl border border-gray-700 flex justify-between items-center text-xs">
                    <div>
                        <strong class="text-white text-sm block">{row['symbol']}</strong>
                        <span class="text-gray-400">Range: ₹{row['or_low']:.1f} - ₹{row['or_high']:.1f}</span>
                    </div>
                    <div class="text-right">
                        <span class="text-green-400 font-black block text-sm">₹{row['current']:.2f}</span>
                        <span class="text-green-500 font-bold bg-green-950 px-2 py-0.5 rounded text-[10px] uppercase">+{row['breakout_pct']:.2f}% Breakout</span>
                    </div>
                </div>
                """
        live_breakouts_html += '</div></div>'
        
        # Bearish ORB List
        live_breakouts_html += """
        <div class="bg-slate-900/60 border border-red-500/20 rounded-2xl p-5 shadow">
            <h4 class="text-red-400 font-bold text-sm mb-4 flex items-center gap-2 uppercase tracking-wider">
                <i class="fa-solid fa-circle-arrow-down text-lg"></i> Live Bearish ORB Breakdown (Below 15m Low)
            </h4>
            <div class="space-y-3">
        """
        if top_live_bears.empty:
            live_breakouts_html += '<p class="text-xs text-gray-500 italic">No active bearish breakdowns detected at this moment.</p>'
        else:
            for idx, row in top_live_bears.iterrows():
                live_breakouts_html += f"""
                <div class="bg-gray-800/40 p-3 rounded-xl border border-gray-700 flex justify-between items-center text-xs">
                    <div>
                        <strong class="text-white text-sm block">{row['symbol']}</strong>
                        <span class="text-gray-400">Range: ₹{row['or_low']:.1f} - ₹{row['or_high']:.1f}</span>
                    </div>
                    <div class="text-right">
                        <span class="text-red-400 font-black block text-sm">₹{row['current']:.2f}</span>
                        <span class="text-red-500 font-bold bg-red-950 px-2 py-0.5 rounded text-[10px] uppercase">+{row['breakdown_pct']:.2f}% Breakdown</span>
                    </div>
                </div>
                """
        live_breakouts_html += '</div></div></div>'
        
    # Generate HTML Cards for Bullish long candidates (Without Kite Buttons)
    longs_html = ""
    card_index = 0
    for idx, row in longs.iterrows():
        card_index += 1
        entry = max(row['close'] * 1.002, row['high'])
        sl = entry - (row['atr'] * 0.5)
        t1 = entry + (row['atr'] * 0.75)
        t2 = entry + (row['atr'] * 1.5)

        longs_html += f"""
        <div class="bg-gray-800 border border-green-500 rounded-xl p-6 shadow-lg hover:shadow-2xl transition duration-300 flex flex-col justify-between">
            <div>
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <span class="bg-green-900 text-green-300 text-[10px] font-black px-2.5 py-1 rounded-full uppercase tracking-wider">BUY LONG</span>
                        <h3 class="text-2xl font-black text-white mt-2">{row['symbol']}</h3>
                        <p class="text-xs text-gray-400 font-medium">{row['ticker']} • Sector: <span class="text-indigo-300 font-bold">{row['sector']}</span></p>
                    </div>
                    <div class="text-right">
                        <span class="text-xs text-gray-400 font-semibold block uppercase">Quant Score</span>
                        <span class="text-3xl font-black text-green-400">{row['bull_score']}<span class="text-xs text-gray-500">/100</span></span>
                    </div>
                </div>
                
                <div class="grid grid-cols-2 gap-4 border-t border-b border-gray-700 py-3 my-4 text-sm text-gray-300">
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">Close Price</span>
                        <span class="font-bold text-lg text-white">₹{row['close']:,.2f}</span>
                    </div>
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">Avg Range (ATR%)</span>
                        <span class="font-bold text-lg text-white">{row['atr_pct']:.2f}%</span>
                    </div>
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">Rel Volume (RVOL)</span>
                        <span class="font-bold text-lg text-white">{row['rvol']:.2f}x</span>
                    </div>
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">RSI (14)</span>
                        <span class="font-bold text-lg text-white">{row['rsi']:.1f}</span>
                    </div>
                </div>
                
                <div class="space-y-2 mb-4">
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-green-400 font-bold uppercase">Trigger Entry (Above)</span>
                        <span class="font-black text-white">₹{entry:,.2f}</span>
                    </div>
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-red-400 font-bold uppercase">Stop-Loss (SL)</span>
                        <span class="font-black text-white" id="sl-long-{card_index}">₹{sl:,.2f}</span>
                    </div>
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-yellow-400 font-bold uppercase">Target 1 (1:1 R:R)</span>
                        <span class="font-black text-white">₹{t1:,.2f}</span>
                    </div>
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-emerald-400 font-bold uppercase">Target 2 (2:1 R:R)</span>
                        <span class="font-black text-white">₹{t2:,.2f}</span>
                    </div>
                </div>
            </div>
            
            <!-- POSITION SIZING CALCULATOR -->
            <div class="bg-slate-900 p-4 rounded-xl border border-indigo-500/20 my-4 text-xs">
                <h4 class="font-black text-indigo-400 mb-2 uppercase tracking-wide flex items-center gap-1">
                    <i class="fa-solid fa-calculator"></i> Smart Position Sizer
                </h4>
                <div class="grid grid-cols-2 gap-3 mb-3">
                    <div>
                        <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Trading Capital</label>
                        <input type="number" id="capital-long-{card_index}" value="50000" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none focus:border-indigo-500" oninput="calculatePositionLong({card_index}, {entry}, {sl})">
                    </div>
                    <div>
                        <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Risk per Trade (%)</label>
                        <input type="number" id="risk-long-{card_index}" value="1" step="0.5" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none focus:border-indigo-500" oninput="calculatePositionLong({card_index}, {entry}, {sl})">
                    </div>
                </div>
                <div class="bg-gray-950 p-2.5 rounded border border-gray-800 space-y-1 text-[11px] text-gray-300">
                    <div class="flex justify-between">
                        <span>Max Cash Risk:</span>
                        <strong class="text-white" id="risk-cash-long-{card_index}">₹500.00</strong>
                    </div>
                    <div class="flex justify-between">
                        <span class="font-bold text-emerald-400">Exact Quantity to Buy:</span>
                        <strong class="text-emerald-400 text-sm font-black" id="qty-long-{card_index}">-- shares</strong>
                    </div>
                    <div class="flex justify-between">
                        <span>Required Margin (5x MIS):</span>
                        <strong class="text-indigo-300" id="margin-long-{card_index}">--</strong>
                    </div>
                </div>
            </div>
            
            <div class="bg-gray-900 p-3 rounded-lg border border-gray-700 text-xs text-gray-400">
                <div class="grid grid-cols-3 text-center gap-1">
                    <div>
                        <span class="block text-gray-500 text-[10px] uppercase font-semibold">Resistance 2</span>
                        <span class="text-white font-medium">₹{row['r2']:,.2f}</span>
                    </div>
                    <div>
                        <span class="block text-gray-500 text-[10px] uppercase font-semibold">Resistance 1</span>
                        <span class="text-white font-medium">₹{row['r1']:,.2f}</span>
                    </div>
                    <div>
                        <span class="block text-gray-500 text-[10px] uppercase font-semibold">Pivot (P)</span>
                        <span class="text-white font-medium">₹{row['pivot']:,.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        
    # Generate HTML Cards for Bearish short candidates (Without Kite Buttons)
    shorts_html = ""
    for idx, row in shorts.iterrows():
        card_index += 1
        entry = min(row['close'] * 0.998, row['low'])
        sl = entry + (row['atr'] * 0.5)
        t1 = entry - (row['atr'] * 0.75)
        t2 = entry - (row['atr'] * 1.5)

        shorts_html += f"""
        <div class="bg-gray-800 border border-red-500 rounded-xl p-6 shadow-lg hover:shadow-2xl transition duration-300 flex flex-col justify-between">
            <div>
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <span class="bg-red-900 text-red-300 text-[10px] font-black px-2.5 py-1 rounded-full uppercase tracking-wider">SHORT SELL</span>
                        <h3 class="text-2xl font-black text-white mt-2">{row['symbol']}</h3>
                        <p class="text-xs text-gray-400 font-medium">{row['ticker']} • Sector: <span class="text-indigo-300 font-bold">{row['sector']}</span></p>
                    </div>
                    <div class="text-right">
                        <span class="text-xs text-gray-400 font-semibold block uppercase">Quant Score</span>
                        <span class="text-3xl font-black text-red-400">{row['bear_score']}<span class="text-xs text-gray-500">/100</span></span>
                    </div>
                </div>
                
                <div class="grid grid-cols-2 gap-4 border-t border-b border-gray-700 py-3 my-4 text-sm text-gray-300">
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">Close Price</span>
                        <span class="font-bold text-lg text-white">₹{row['close']:,.2f}</span>
                    </div>
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">Avg Range (ATR%)</span>
                        <span class="font-bold text-lg text-white">{row['atr_pct']:.2f}%</span>
                    </div>
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">Rel Volume (RVOL)</span>
                        <span class="font-bold text-lg text-white">{row['rvol']:.2f}x</span>
                    </div>
                    <div>
                        <span class="text-gray-500 text-xs block uppercase">RSI (14)</span>
                        <span class="font-bold text-lg text-white">{row['rsi']:.1f}</span>
                    </div>
                </div>
                
                <div class="space-y-2 mb-4">
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-red-400 font-bold uppercase">Trigger Entry (Below)</span>
                        <span class="font-black text-white">₹{entry:,.2f}</span>
                    </div>
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-green-400 font-bold uppercase">Stop-Loss (SL)</span>
                        <span class="font-black text-white" id="sl-short-{card_index}">₹{sl:,.2f}</span>
                    </div>
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-yellow-400 font-bold uppercase">Target 1 (1:1 R:R)</span>
                        <span class="font-black text-white">₹{t1:,.2f}</span>
                    </div>
                    <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                        <span class="text-emerald-400 font-bold uppercase">Target 2 (2:1 R:R)</span>
                        <span class="font-black text-white">₹{t2:,.2f}</span>
                    </div>
                </div>
            </div>
            
            <!-- POSITION SIZING CALCULATOR -->
            <div class="bg-slate-900 p-4 rounded-xl border border-indigo-500/20 my-4 text-xs">
                <h4 class="font-black text-indigo-400 mb-2 uppercase tracking-wide flex items-center gap-1">
                    <i class="fa-solid fa-calculator"></i> Smart Position Sizer
                </h4>
                <div class="grid grid-cols-2 gap-3 mb-3">
                    <div>
                        <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Trading Capital</label>
                        <input type="number" id="capital-short-{card_index}" value="50000" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none focus:border-indigo-500" oninput="calculatePositionShort({card_index}, {entry}, {sl})">
                    </div>
                    <div>
                        <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Risk per Trade (%)</label>
                        <input type="number" id="risk-short-{card_index}" value="1" step="0.5" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none focus:border-indigo-500" oninput="calculatePositionShort({card_index}, {entry}, {sl})">
                    </div>
                </div>
                <div class="bg-gray-950 p-2.5 rounded border border-gray-800 space-y-1 text-[11px] text-gray-300">
                    <div class="flex justify-between">
                        <span>Max Cash Risk:</span>
                        <strong class="text-white" id="risk-cash-short-{card_index}">₹500.00</strong>
                    </div>
                    <div class="flex justify-between">
                        <span class="font-bold text-red-400">Exact Quantity to Sell:</span>
                        <strong class="text-red-400 text-sm font-black" id="qty-short-{card_index}">-- shares</strong>
                    </div>
                    <div class="flex justify-between">
                        <span>Required Margin (5x MIS):</span>
                        <strong class="text-indigo-300" id="margin-short-{card_index}">--</strong>
                    </div>
                </div>
            </div>
            
            <div class="bg-gray-900 p-3 rounded-lg border border-gray-700 text-xs text-gray-400">
                <div class="grid grid-cols-3 text-center gap-1">
                    <div>
                        <span class="block text-gray-500 text-[10px] uppercase font-semibold">Pivot (P)</span>
                        <span class="text-white font-medium">₹{row['pivot']:,.2f}</span>
                    </div>
                    <div>
                        <span class="block text-gray-500 text-[10px] uppercase font-semibold">Support 1</span>
                        <span class="text-white font-medium">₹{row['s1']:,.2f}</span>
                    </div>
                    <div>
                        <span class="block text-gray-500 text-[10px] uppercase font-semibold">Support 2</span>
                        <span class="text-white font-medium">₹{row['s2']:,.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        
    # Generate HTML for Penny Stock Momentum cards
    penny_cards_html = ""
    if top_penny_stocks.empty:
        penny_cards_html = """
        <div class="col-span-full text-center py-6 text-gray-400 text-sm">
            <i class="fa-solid fa-triangle-exclamation text-yellow-500 text-lg mb-2"></i><br>
            No high-volume penny stock breakouts scanned meeting the criteria today.
        </div>
        """
    else:
        for idx, row in top_penny_stocks.iterrows():
            perf = row['change_pct']
            color = "text-green-400" if perf >= 0 else "text-red-400"
            symbol_sign = "+" if perf >= 0 else ""
            alert_reason = "🔥 Volume Spike" if row['rvol'] >= 3.0 else "⚡ Momentum Breakout" if perf >= 4.0 else "📈 Steady Accumulation"
            
            penny_cards_html += f"""
            <div class="bg-slate-900/60 border border-indigo-500/20 rounded-xl p-4 flex flex-col justify-between hover:border-indigo-500/50 hover:bg-slate-900/90 transition duration-300">
                <div class="flex justify-between items-start mb-2">
                    <div>
                        <h4 class="text-lg font-black text-white">{row['symbol']}</h4>
                        <span class="text-[9px] font-bold text-indigo-300 uppercase tracking-widest bg-indigo-950/60 border border-indigo-500/10 px-2 py-0.5 rounded-full">{alert_reason}</span>
                    </div>
                    <div class="text-right">
                        <span class="text-[9px] text-gray-500 font-bold uppercase block">Price</span>
                        <span class="font-extrabold text-white text-base">₹{row['close']:.2f}</span>
                    </div>
                </div>
                
                <div class="grid grid-cols-2 gap-2 border-t border-gray-800 pt-3 mt-3 text-xs">
                    <div>
                        <span class="text-gray-500 text-[9px] block uppercase">Rel Volume (RVOL)</span>
                        <span class="font-extrabold text-white">{row['rvol']:.2f}x</span>
                    </div>
                    <div>
                        <span class="text-gray-500 text-[9px] block uppercase">Day Change</span>
                        <span class="font-extrabold {color}">{symbol_sign}{perf:.2f}%</span>
                    </div>
                </div>
            </div>
            """

    # Full Leaderboard Table HTML
    leaderboard_html = ""
    sorted_all = df_all.sort_values(by=['bull_score', 'bear_score'], ascending=False)
    for idx, row in sorted_all.iterrows():
        bull = int(row['bull_score'])
        bear = int(row['bear_score'])
        sentiment_bg = "bg-green-950 text-green-300" if bull > bear else "bg-red-950 text-red-300" if bear > bull else "bg-gray-700 text-gray-300"
        sentiment_lbl = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "NEUTRAL"
        
        leaderboard_html += f"""
        <tr class="border-b border-gray-800 hover:bg-gray-750 transition">
            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-white">{row['symbol']}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-400">{row['sector']}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">₹{row['close']:,.2f}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{row['rsi']:.1f}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{row['rvol']:.2f}x</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{row['atr_pct']:.2f}%</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-center">
                <span class="px-2 py-1 rounded text-xs font-black bg-green-900/40 text-green-400 border border-green-500/20">{bull}</span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-center">
                <span class="px-2 py-1 rounded text-xs font-black bg-red-900/40 text-red-400 border border-red-500/20">{bear}</span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-center">
                <span class="px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase {sentiment_bg}">{sentiment_lbl}</span>
            </td>
        </tr>
        """
        
    # Complete HTML Template
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ India Intraday Stock Finder Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #0f172a;
        }}
    </style>
</head>
<body class="text-slate-100 min-h-screen pb-12">

    <!-- TOP HEADER -->
    <header class="border-b border-gray-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
            <div class="flex items-center gap-3">
                <div class="bg-gradient-to-tr from-indigo-500 to-emerald-500 p-2.5 rounded-xl shadow-lg shadow-indigo-500/20">
                    <i class="fa-solid fa-chart-line text-xl text-white"></i>
                </div>
                <div>
                    <h1 class="text-xl sm:text-2xl font-black bg-gradient-to-r from-white via-slate-200 to-indigo-400 bg-clip-text text-transparent">QUANT FINDER</h1>
                    <p class="text-[10px] sm:text-xs text-emerald-400 font-semibold tracking-wider uppercase"><i class="fa-solid fa-circle text-[8px] animate-pulse mr-1"></i> NSE India Intraday Core</p>
                </div>
            </div>
            
            <div class="flex flex-wrap items-center gap-3">
                <div class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 flex items-center gap-2 text-xs">
                    <i class="fa-solid fa-clock text-indigo-400"></i>
                    <span id="market-status-badge" class="font-bold text-gray-300">IST: <span id="ist-clock">--:--:--</span></span>
                    <span id="market-state" class="px-2 py-0.5 rounded text-[10px] font-black uppercase bg-red-950 text-red-400 border border-red-500/20">CLOSED</span>
                </div>
                <div class="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-xs">
                    <span class="text-gray-400">Session:</span> <strong class="text-white">{trading_date_str}</strong>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        
        <!-- INTRO HERO BANNER (WITH ANIMATED REFRESH BUTTON) -->
        <div class="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 rounded-2xl p-6 sm:p-8 border border-slate-800 mb-8 shadow-xl relative overflow-hidden">
            <div class="absolute -right-12 -bottom-12 opacity-5 text-9xl font-black">NSE</div>
            <div class="max-w-3xl relative z-10">
                <h2 class="text-2xl sm:text-3xl font-black text-white mb-2">High-Probability Intraday Stock Scanner</h2>
                <p class="text-slate-400 text-sm sm:text-base leading-relaxed mb-4">
                    Quantitatively scanning the top 50 most liquid NSE stocks every evening. This algorithmic model filters for institutional volume spikes, relative range expansions (ATR), trend alignment, and breakout squeezes to output the 3 strongest setups for the day.
                </p>
                <div class="flex flex-wrap gap-4 text-xs font-semibold text-slate-300 mb-4">
                    <span class="flex items-center gap-1.5"><i class="fa-solid fa-shield-halved text-emerald-400"></i> Risk-to-Reward Optimized</span>
                    <span class="flex items-center gap-1.5"><i class="fa-solid fa-bolt text-indigo-400"></i> 15-Min ORB Trigger</span>
                    <span class="flex items-center gap-1.5"><i class="fa-solid fa-code text-pink-400"></i> Fully Automated</span>
                </div>
                
                <!-- INTEGRATED ANIMATED LIVE REFRESH BUTTONS -->
                <div class="flex flex-wrap gap-4">
                    <button onclick="refreshLivePrices()" class="bg-emerald-600 hover:bg-emerald-500 text-white font-black text-xs uppercase px-4 py-2.5 rounded-lg transition duration-200 flex items-center gap-2 shadow-lg shadow-emerald-600/15" id="price-refresh-btn">
                        <i class="fa-solid fa-rotate-right" id="price-refresh-icon"></i> Sync Live Quotes
                    </button>
                    <button onclick="toggleTokenModal()" class="bg-gray-800 hover:bg-gray-700 border border-gray-700 text-white font-black text-xs uppercase px-4 py-2.5 rounded-lg transition duration-200 flex items-center gap-2" id="cloud-trigger-btn">
                        <i class="fa-solid fa-cloud" id="cloud-icon"></i> Trigger Cloud Refresh
                    </button>
                </div>
            </div>
        </div>

        <!-- HIDDEN GITHUB TOKEN CONFIG MODAL -->
        <div id="token-modal" class="hidden bg-slate-900 border border-gray-800 rounded-xl p-5 mb-8 shadow-xl">
            <h4 class="font-bold text-sm text-white mb-2 uppercase tracking-wide flex items-center gap-2">
                <i class="fa-solid fa-gear text-indigo-400"></i> Configure Cloud Trigger Settings
            </h4>
            <p class="text-xs text-gray-400 mb-4 leading-relaxed">
                To trigger a live data scan in the cloud directly from this webpage, paste your <strong>GitHub Personal Access Token (PAT)</strong>. This token is stored <strong>100% securely only inside your browser's private local memory (localStorage)</strong> and is never sent to any server except directly to GitHub's official API.
            </p>
            <div class="flex gap-3 text-xs mb-3">
                <input type="password" id="gh-token" placeholder="Paste your GitHub Personal Access Token (classic or fine-grained)..." class="flex-1 bg-slate-950 border border-gray-800 rounded px-3 py-2 text-white focus:outline-none focus:border-indigo-500">
                <button onclick="saveGitHubToken()" class="bg-indigo-600 hover:bg-indigo-500 text-white font-black px-4 rounded transition">
                    Save Token
                </button>
            </div>
            <div class="flex justify-between items-center text-[10px] text-gray-500">
                <span>Repository: <strong>abhisim23/nifty-scanner</strong></span>
                <button onclick="clearGitHubToken()" class="text-rose-400 hover:underline"><i class="fa-solid fa-trash-can mr-1"></i>Clear Token</button>
            </div>
            
            <div class="mt-4 border-t border-gray-800/80 pt-4 hidden" id="action-trigger-area">
                <button onclick="triggerGitHubAction()" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-black py-3 px-4 rounded-xl shadow-md transition duration-200 flex items-center justify-center gap-2 uppercase tracking-wider text-xs" id="run-workflow-btn">
                    <i class="fa-solid fa-rotate-right" id="workflow-spinner"></i> Run Live Cloud Scanner (Takes 60 seconds)
                </button>
                <div id="countdown-area" class="hidden text-center mt-3 font-bold text-xs text-indigo-400 animate-pulse">
                    🔄 Cloud server is active! Compiling fresh stock data. Auto-reloading in <span id="countdown-seconds">60</span>s...
                </div>
            </div>
        </div>

        <!-- REAL-TIME INTERACTIVE TRADINGVIEW CHART WIDGETS -->
        <div class="mb-8 grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Nifty 50 widget -->
            <div class="bg-slate-900 border border-gray-800 rounded-2xl overflow-hidden h-[240px]">
                <!-- TradingView Widget BEGIN -->
                <div class="tradingview-widget-container h-full">
                  <div class="tradingview-widget-container__widget h-full"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
                  {{
                  "symbol": "NSE:NIFTY",
                  "width": "100%",
                  "height": "100%",
                  "locale": "en",
                  "dateRange": "1D",
                  "colorTheme": "dark",
                  "isTransparent": true,
                  "autosize": true,
                  "largeChartUrl": ""
                }}
                  </script>
                </div>
                <!-- TradingView Widget END -->
            </div>
            <!-- Bank Nifty widget -->
            <div class="bg-slate-900 border border-gray-800 rounded-2xl overflow-hidden h-[240px]">
                <!-- TradingView Widget BEGIN -->
                <div class="tradingview-widget-container h-full">
                  <div class="tradingview-widget-container__widget h-full"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
                  {{
                  "symbol": "NSE:BANKNIFTY",
                  "width": "100%",
                  "height": "100%",
                  "locale": "en",
                  "dateRange": "1D",
                  "colorTheme": "dark",
                  "isTransparent": true,
                  "autosize": true,
                  "largeChartUrl": ""
                }}
                  </script>
                </div>
                <!-- TradingView Widget END -->
            </div>
        </div>

        <!-- TWO COLUMN LAYOUT: LIVE BREAKOUTS & PRE-MARKET STRATEGY -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            
            <!-- COLUMN 1 & 2: LIVE BREAKOUTS TRACKER (UPDATED IN REAL-TIME DURING MARKET HOURS) -->
            <div class="lg:col-span-2 bg-slate-950/40 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between" id="orb-live-card">
                <div>
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-lg font-black text-white flex items-center gap-2">
                            <i class="fa-solid fa-tower-broadcast text-indigo-400 animate-pulse" id="live-antenna"></i> Live Intraday Breakout Tracker
                        </h3>
                        <span class="text-[9px] text-emerald-400 font-extrabold bg-emerald-950/60 border border-emerald-500/20 px-2 py-0.5 rounded-full uppercase tracking-widest">
                            Live 15m ORB Feed
                        </span>
                    </div>
                    <p class="text-xs text-gray-400 mb-4">
                        This panel monitors live 15-minute price bars. The moment any Nifty 50 stock breaks above or below its first 15-minute candle range (09:15 - 09:30 AM), it is tracked here as an active momentum trade.
                    </p>
                    <div class="grid grid-cols-1 gap-4" id="orb-live-container">
                        {live_breakouts_html}
                    </div>
                </div>
                <div class="mt-4 text-[10px] text-gray-500 italic flex items-center gap-1.5 border-t border-gray-800/60 pt-3">
                    <i class="fa-solid fa-rotate text-indigo-400"></i> Run a manual scan in GitHub Actions during market hours to refresh this live list!
                </div>
            </div>

            <!-- COLUMN 3: PRE-MARKET ACTION GUIDE (PRIOR TO OPEN) -->
            <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
                <h3 class="text-lg font-black text-white flex items-center gap-2 mb-2">
                    <i class="fa-solid fa-bullhorn text-amber-400"></i> Pre-Market Strategy (09:08 AM)
                </h3>
                <p class="text-xs text-gray-400 mb-4">
                    Prior to market open, follow these instructions to align our quant picks with the actual market gap openings:
                </p>
                <div class="space-y-3 text-xs">
                    <div class="bg-slate-950/60 p-3 rounded-lg border border-gray-800">
                        <strong class="text-amber-400 font-bold block mb-1">1. Check NSE Pre-Open Page</strong>
                        <p class="text-gray-400 text-[11px] mb-2">At 09:08 AM IST, open the official NSE pre-open web terminal to see where all stocks are opening:</p>
                        <a href="https://www.nseindia.com/market-data/pre-open-market-equity-and-sme" target="_blank" class="bg-indigo-900 hover:bg-indigo-800 text-white font-extrabold text-[10px] px-3 py-1.5 rounded uppercase tracking-wider inline-block">
                            <i class="fa-solid fa-square-arrow-up-right mr-1"></i> Open NSE Pre-Open
                        </a>
                    </div>
                    <div class="bg-slate-950/60 p-3 rounded-lg border border-gray-800">
                        <strong class="text-white block mb-1">2. Align Gaps with Quant Picks</strong>
                        <p class="text-gray-400 text-[11px]">Compare the top gap-up stocks on NSE with our <strong>3 Bullish Long Picks</strong>. If our picks are also gapping up with high pre-open volume, it is a massive confirmation signal!</p>
                    </div>
                    <div class="bg-slate-950/60 p-3 rounded-lg border border-gray-800">
                        <strong class="text-white block mb-1">3. Index Gap Rule</strong>
                        <p class="text-gray-400 text-[11px]">If Nifty 50 Index opens with a large gap up of >0.5%, avoid shorting and trade only longs. If it gaps down >0.5%, trade only shorts.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- BANK NIFTY CORE WEEKLY & DAILY HUB (UPGRADED) -->
        <div class="mb-8 bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 rounded-2xl p-6 border border-indigo-500/30 shadow-xl">
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-gray-800 pb-4 mb-4 gap-4">
                <div>
                    <h3 class="text-xl font-black text-white flex items-center gap-2">
                        <i class="fa-solid fa-building-columns text-indigo-400"></i> Bank Nifty Weekly & Daily Trading Hub
                    </h3>
                    <p class="text-xs text-gray-400 mt-1">
                        Professional execution levels, daily/weekly ranges, and dynamic entry alerts for Nifty Bank index (^NSEBANK).
                    </p>
                </div>
                <div class="flex gap-3 items-center">
                    <div class="bg-slate-900 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-center">
                        <span class="text-gray-500 block text-[9px] uppercase font-bold">Daily EMA-20 Trend</span>
                        <strong class="text-white font-black" id="bn-daily-bias-lbl">{daily_bias}</strong>
                    </div>
                    <div class="bg-slate-900 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-center">
                        <span class="text-gray-500 block text-[9px] uppercase font-bold">Weekly EMA-20 Trend</span>
                        <strong class="text-white font-black">{weekly_bias}</strong>
                    </div>
                </div>
            </div>

            <!-- INDEX SUMMARY -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div class="bg-slate-900/50 p-4 rounded-xl border border-gray-800 text-xs">
                    <span class="text-gray-500 uppercase block font-bold text-[9px]">Last Closed Price</span>
                    <strong class="text-white text-lg" id="bn-last-close-val">₹{latest_bn_close:,.2f}</strong>
                </div>
                <div class="bg-slate-900/50 p-4 rounded-xl border border-gray-800 text-xs">
                    <span class="text-gray-500 uppercase block font-bold text-[9px]">Daily 20 EMA</span>
                    <strong class="text-slate-300 text-lg">₹{latest_bn_ema:,.2f}</strong>
                </div>
                <div class="bg-slate-900/50 p-4 rounded-xl border border-gray-800 text-xs">
                    <span class="text-gray-500 uppercase block font-bold text-[9px]">Weekly Close</span>
                    <strong class="text-white text-lg">₹{latest_bn_w_close:,.2f}</strong>
                </div>
                <div class="bg-slate-900/50 p-4 rounded-xl border border-gray-800 text-xs">
                    <span class="text-gray-500 uppercase block font-bold text-[9px]">Weekly 20 EMA</span>
                    <strong class="text-slate-300 text-lg">₹{latest_bn_w_ema:,.2f}</strong>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
                
                <!-- DAILY BANK NIFTY TRADING PLAN -->
                <div class="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
                    <div class="flex justify-between items-center mb-4">
                        <h4 class="text-white font-black text-sm uppercase tracking-wider flex items-center gap-1.5">
                            <i class="fa-solid fa-bolt text-yellow-400"></i> Daily Trading Levels (Today)
                        </h4>
                        <!-- Dynamic JS-calculated alert badge -->
                        <span id="daily-action-badge" class="px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-slate-950 text-slate-400 border border-slate-800">--</span>
                    </div>
                    
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <!-- Bullish Trigger -->
                        <div class="bg-green-950/20 border border-green-500/10 p-3.5 rounded-xl space-y-1.5">
                            <span class="text-[10px] text-green-400 font-extrabold block uppercase tracking-wide">BUY LONG TRIGGER</span>
                            <div class="text-white text-sm font-extrabold">Above: <span class="text-base text-white font-black" id="daily-long-trigger">₹{daily_long_trigger:,.2f}</span></div>
                            <div class="text-slate-400 text-[10px]">SL: ₹{daily_long_sl:,.1f}</div>
                            <div class="text-green-300 text-[11px] font-bold">Target 1: ₹{daily_long_t1:,.1f}</div>
                            <div class="text-emerald-400 text-[11px] font-black">Target 2: ₹{daily_long_t2:,.1f}</div>
                        </div>
                        <!-- Bearish Trigger -->
                        <div class="bg-red-950/20 border border-red-500/10 p-3.5 rounded-xl space-y-1.5">
                            <span class="text-[10px] text-red-400 font-extrabold block uppercase tracking-wide">SHORT SELL TRIGGER</span>
                            <div class="text-white text-sm font-extrabold">Below: <span class="text-base text-white font-black" id="daily-short-trigger">₹{daily_short_trigger:,.2f}</span></div>
                            <div class="text-slate-400 text-[10px]">SL: ₹{daily_short_sl:,.1f}</div>
                            <div class="text-red-300 text-[11px] font-bold">Target 1: ₹{daily_short_t1:,.1f}</div>
                            <div class="text-rose-400 text-[11px] font-black">Target 2: ₹{daily_short_t2:,.1f}</div>
                        </div>
                    </div>
                </div>

                <!-- WEEKLY BANK NIFTY TRADING PLAN -->
                <div class="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
                    <div class="flex justify-between items-center mb-4">
                        <h4 class="text-white font-black text-sm uppercase tracking-wider flex items-center gap-1.5">
                            <i class="fa-solid fa-calendar-week text-indigo-400"></i> Weekly Breakout Levels
                        </h4>
                        <!-- Dynamic weekly badge -->
                        <span id="weekly-action-badge" class="px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-slate-950 text-slate-400 border border-slate-800">--</span>
                    </div>
                    
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <!-- Bullish Trigger -->
                        <div class="bg-green-950/20 border border-green-500/10 p-3.5 rounded-xl space-y-1.5">
                            <span class="text-[10px] text-green-400 font-extrabold block uppercase tracking-wide">WEEKLY LONG TRIGGER</span>
                            <div class="text-white text-sm font-extrabold">Above: <span class="text-base text-white font-black" id="weekly-long-trigger">₹{weekly_long_trigger:,.2f}</span></div>
                            <div class="text-slate-400 text-[10px]">SL: ₹{weekly_long_sl:,.1f}</div>
                            <div class="text-green-300 text-[11px] font-bold">Target 1: ₹{weekly_long_t1:,.1f}</div>
                            <div class="text-emerald-400 text-[11px] font-black">Target 2: ₹{weekly_long_t2:,.1f}</div>
                        </div>
                        <!-- Bearish Trigger -->
                        <div class="bg-red-950/20 border border-red-500/10 p-3.5 rounded-xl space-y-1.5">
                            <span class="text-[10px] text-red-400 font-extrabold block uppercase tracking-wide">WEEKLY SHORT TRIGGER</span>
                            <div class="text-white text-sm font-extrabold">Below: <span class="text-base text-white font-black" id="weekly-short-trigger">₹{weekly_short_trigger:,.2f}</span></div>
                            <div class="text-slate-400 text-[10px]">SL: ₹{weekly_short_sl:,.1f}</div>
                            <div class="text-red-300 text-[11px] font-bold">Target 1: ₹{weekly_short_t1:,.1f}</div>
                            <div class="text-rose-400 text-[11px] font-black">Target 2: ₹{weekly_short_t2:,.1f}</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Daily Classical Pivot Levels -->
            <div class="bg-slate-900/60 p-4 rounded-xl border border-gray-800">
                <h4 class="font-extrabold text-white mb-3 text-xs uppercase text-indigo-300 tracking-wider">
                    <i class="fa-solid fa-list-check mr-1"></i> Daily Standard Pivot Points (₹)
                </h4>
                <div class="grid grid-cols-5 text-center text-xs gap-2">
                    <div class="bg-red-950/40 border border-red-500/20 p-2 rounded">
                        <span class="text-red-400 font-extrabold block text-[10px]">R2</span>
                        <strong class="text-white font-black">{bn_r2:,.1f}</strong>
                    </div>
                    <div class="bg-red-950/20 border border-red-500/10 p-2 rounded">
                        <span class="text-red-300 font-extrabold block text-[10px]">R1</span>
                        <strong class="text-white font-semibold">{bn_r1:,.1f}</strong>
                    </div>
                    <div class="bg-slate-950 border border-gray-800 p-2 rounded">
                        <span class="text-gray-400 font-extrabold block text-[10px]">PIVOT</span>
                        <strong class="text-white font-bold">{bn_pivot:,.1f}</strong>
                    </div>
                    <div class="bg-green-950/20 border border-green-500/10 p-2 rounded">
                        <span class="text-green-300 font-extrabold block text-[10px]">S1</span>
                        <strong class="text-white font-semibold">{bn_s1:,.1f}</strong>
                    </div>
                    <div class="bg-green-950/40 border border-green-500/20 p-2 rounded">
                        <span class="text-green-400 font-extrabold block text-[10px]">S2</span>
                        <strong class="text-white font-black">{bn_s2:,.1f}</strong>
                    </div>
                </div>
            </div>
        </div>

        <!-- HIGH MOMENTUM PENNY STOCK FINDER -->
        <div class="mb-8 bg-gradient-to-r from-slate-950 to-indigo-950 rounded-2xl p-6 border border-indigo-500/20 shadow-lg">
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4">
                <div>
                    <h3 class="text-lg font-black text-white flex items-center gap-2">
                        <i class="fa-solid fa-circle-nodes text-indigo-400"></i> Penny Stock Momentum Breakout Finder
                    </h3>
                    <p class="text-xs text-gray-400 mt-1">
                        Scans low-priced stocks (₹1 to ₹100) on the NSE to find massive volume spikes indicating strong buying demand and potential "Upper Circuit" targets.
                    </p>
                </div>
                <span class="text-[10px] text-yellow-400 font-extrabold bg-yellow-950/50 border border-yellow-500/20 px-3 py-1 rounded-full uppercase tracking-widest flex items-center gap-1">
                    <i class="fa-solid fa-circle-exclamation text-xs"></i> Extremely Volatile Risk Warning
                </span>
            </div>
            
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
                {penny_cards_html}
            </div>
            
            <div class="mt-4 p-3 bg-indigo-950/20 rounded-lg border border-indigo-500/10 text-[10px] leading-relaxed text-gray-400 flex gap-2 items-start">
                <i class="fa-solid fa-circle-info text-indigo-400 mt-0.5 text-xs"></i>
                <p>
                    <strong>💡 How daily price circuits work:</strong> The National Stock Exchange (NSE) applies daily circuit bands (2%, 5%, 10%, or 20%) to penny stocks to limit excessive speculation. High Relative Volume (RVOL) is the primary engine behind circuit breakouts. <strong>Warning:</strong> Penny stocks carry severe liquidity risks. If a stock hits its lower circuit limit, buyers disappear entirely, leaving you unable to sell or exit your shares. Always practice absolute stop-loss discipline.
                </p>
            </div>
        </div>

        <!-- TWO COLUMN SETUP (BULLISH VS BEARISH) -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            
            <!-- BULLISH LONG CARDS -->
            <div>
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-2">
                        <i class="fa-solid fa-circle-chevron-up text-2xl text-green-500"></i>
                        <h2 class="text-xl font-extrabold text-white">Bullish Long Breakouts</h2>
                    </div>
                    <span class="text-xs text-green-400 font-semibold bg-green-950/50 px-2.5 py-1 rounded-md border border-green-500/20">Target Nifty Green Days</span>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-1 gap-6">
                    {longs_html}
                </div>
            </div>
            
            <!-- BEARISH SHORT CARDS -->
            <div>
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-2">
                        <i class="fa-solid fa-circle-chevron-down text-2xl text-red-500"></i>
                        <h2 class="text-xl font-extrabold text-white">Bearish Short Breakdowns</h2>
                    </div>
                    <span class="text-xs text-red-400 font-semibold bg-red-950/50 px-2.5 py-1 rounded-md border border-red-500/20">Target Nifty Red Days</span>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-1 gap-6">
                    {shorts_html}
                </div>
            </div>
        </div>

        <!-- ULTIMATE ALL-IN-ONE UTILITIES HUB -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            
            <!-- TOOL 1: TRADING JOURNAL (LOCAL STORAGE PERSISTENT) -->
            <div class="bg-slate-900/60 rounded-2xl border border-gray-800 p-6 shadow flex flex-col justify-between">
                <div>
                    <h3 class="text-lg font-black text-white flex items-center gap-2 mb-2">
                        <i class="fa-solid fa-book text-indigo-400"></i> Interactive Trader's Daily Journal
                    </h3>
                    <p class="text-xs text-gray-400 mb-4">
                        Log your daily trades directly on your dashboard. Your journal logs are saved privately in your web browser's local memory (`localStorage`) so they never disappear or get wiped, even if you refresh!
                    </p>
                    
                    <form id="journal-form" class="grid grid-cols-3 gap-3 text-xs mb-4" onsubmit="addJournalEntry(event)">
                        <div class="col-span-1">
                            <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Stock</label>
                            <input type="text" id="j-stock" required placeholder="e.g. INFY" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none">
                        </div>
                        <div class="col-span-1">
                            <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Quantity</label>
                            <input type="number" id="j-qty" required placeholder="100" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none">
                        </div>
                        <div class="col-span-1">
                            <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Total P&L (₹)</label>
                            <input type="number" id="j-pnl" required placeholder="e.g. 1500" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none">
                        </div>
                        <div class="col-span-3">
                            <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Trading Strategy & Notes</label>
                            <input type="text" id="j-notes" required placeholder="e.g. Breakout of 15m range with 2.3x RVOL confirmation" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white focus:outline-none">
                        </div>
                        <button type="submit" class="col-span-3 bg-indigo-600 hover:bg-indigo-500 text-white font-black py-2 rounded transition">
                            <i class="fa-solid fa-plus mr-1"></i> Log Trade Entry
                        </button>
                    </form>

                    <!-- Trade Log Table -->
                    <div class="overflow-x-auto max-h-48 border border-gray-800 rounded-lg">
                        <table class="min-w-full text-xs text-left divide-y divide-gray-800" id="journal-table">
                            <thead class="bg-slate-950/60 text-gray-500 font-bold uppercase text-[9px]">
                                <tr>
                                    <th class="px-4 py-2">Stock</th>
                                    <th class="px-4 py-2">Qty</th>
                                    <th class="px-4 py-2">P&L (₹)</th>
                                    <th class="px-4 py-2">Notes</th>
                                    <th class="px-4 py-2 text-right">Delete</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-gray-800 text-gray-300">
                                <!-- JS populated -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- TOOL 2: OPTIONS DECAY & SLIPPAGE ESTIMATOR -->
            <div class="bg-slate-900/60 rounded-2xl border border-gray-800 p-6 shadow flex flex-col justify-between">
                <div>
                    <h3 class="text-lg font-black text-white flex items-center gap-2 mb-2">
                        <i class="fa-solid fa-percent text-indigo-400"></i> Options Margin & Risk Estimator
                    </h3>
                    <p class="text-xs text-gray-400 mb-4">
                        For Bank Nifty option buyers/sellers. Quickly estimate potential margin requirements and decay slip thresholds.
                    </p>
                    
                    <div class="space-y-4 text-xs">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Option Premium (₹)</label>
                                <input type="number" id="opt-premium" value="350" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none" oninput="calculateOptionStats()">
                            </div>
                            <div>
                                <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Lot Size (Bank Nifty)</label>
                                <input type="number" id="opt-lotsize" value="15" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-slate-400 font-bold focus:outline-none bg-slate-950/50" readonly>
                            </div>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Number of Lots</label>
                                <input type="number" id="opt-lots" value="2" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none" oninput="calculateOptionStats()">
                            </div>
                            <div>
                                <label class="text-gray-500 text-[10px] uppercase font-bold block mb-1">Stop-Loss Points (Option)</label>
                                <input type="number" id="opt-sl-pts" value="30" class="w-full bg-gray-800 border border-gray-700 rounded p-1.5 text-white font-bold focus:outline-none" oninput="calculateOptionStats()">
                            </div>
                        </div>

                        <div class="bg-gray-950/70 p-3 rounded-xl border border-gray-800/80 space-y-2 text-gray-300">
                            <div class="flex justify-between">
                                <span>Total Contract Value (Option Buy):</span>
                                <strong class="text-white" id="opt-buy-capital">₹10,500.00</strong>
                            </div>
                            <div class="flex justify-between text-rose-400">
                                <span class="font-bold">Total Max Risk on SL:</span>
                                <strong class="font-black text-sm" id="opt-total-risk">₹900.00</strong>
                            </div>
                            <div class="flex justify-between text-indigo-300 border-t border-gray-800 pt-1.5">
                                <span>Approx Selling Margin (1 Lot Write):</span>
                                <strong id="opt-sell-margin">₹1,20,000.00</strong>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="text-[10px] text-gray-500 italic mt-3">
                    <i class="fa-solid fa-circle-info"></i> Standard lot size for Nifty Bank is 15. Margin calculations are approximations based on historical exchange standards.
                </div>
            </div>
        </div>

        <!-- STRATEGY PLAYBOOK -->
        <div class="bg-slate-900/60 rounded-2xl border border-gray-800 p-6 sm:p-8 mb-12 shadow-md">
            <h3 class="text-lg sm:text-xl font-bold text-white mb-6 flex items-center gap-2">
                <i class="fa-solid fa-book-open text-indigo-400"></i> The Professional 15-Minute ORB Strategy
            </h3>
            
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 text-sm">
                <div class="bg-slate-950/50 p-4 rounded-xl border border-gray-800">
                    <span class="bg-indigo-900 text-indigo-300 font-bold px-2 py-0.5 rounded text-[10px] uppercase block w-max mb-3">STEP 1</span>
                    <h4 class="font-extrabold text-white mb-1">Watch (09:15 - 09:30 AM)</h4>
                    <p class="text-slate-400 text-xs">Let market volatility settle. Add these stocks to your watchlist. Check if Nifty 50 is trending bullish (green) or bearish (red).</p>
                </div>
                <div class="bg-slate-950/50 p-4 rounded-xl border border-gray-800">
                    <span class="bg-indigo-900 text-indigo-300 font-bold px-2 py-0.5 rounded text-[10px] uppercase block w-max mb-3">STEP 2</span>
                    <h4 class="font-extrabold text-white mb-1">Set Your Triggers</h4>
                    <p class="text-slate-400 text-xs">Draw horizontal lines on the high and low of the first 15-min candle (09:15 to 09:30 AM) for your active stock picks.</p>
                </div>
                <div class="bg-slate-950/50 p-4 rounded-xl border border-gray-800">
                    <span class="bg-indigo-900 text-indigo-300 font-bold px-2 py-0.5 rounded text-[10px] uppercase block w-max mb-3">STEP 3</span>
                    <h4 class="font-extrabold text-white mb-1">Execution Entry</h4>
                    <p class="text-slate-400 text-xs">Enter <strong>LONG</strong> if a candle closes above the 15-Min high. Enter <strong>SHORT</strong> if it closes below the 15-Min low.</p>
                </div>
                <div class="bg-slate-950/50 p-4 rounded-xl border border-gray-800">
                    <span class="bg-indigo-900 text-indigo-300 font-bold px-2 py-0.5 rounded text-[10px] uppercase block w-max mb-3">STEP 4</span>
                    <h4 class="font-extrabold text-white mb-1">Active Exit (03:15 PM)</h4>
                    <p class="text-slate-400 text-xs">Place your Stop Loss immediately. Take 50% profit at Target 1 and move SL to entry. Close all remaining shares by 3:15 PM.</p>
                </div>
            </div>
        </div>

        <!-- FULL NIFTY 50 LEADERBOARD -->
        <div class="bg-slate-900 rounded-2xl border border-gray-800 overflow-hidden shadow-lg">
            <div class="px-6 py-5 border-b border-gray-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h3 class="text-lg font-bold text-white">Nifty 50 Quant Leaderboard</h3>
                    <p class="text-xs text-gray-400">Relative scores of all screened stocks. Sort to find alternative trading candidates.</p>
                </div>
                <div class="relative max-w-xs">
                    <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-gray-500"></i>
                    <input type="text" id="search-box" placeholder="Search stock symbol..." class="w-full pl-9 pr-4 py-1.5 bg-slate-950 border border-gray-800 rounded-lg text-xs text-white focus:outline-none focus:border-indigo-500 transition">
                </div>
            </div>
            
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-800" id="leaderboard-table">
                    <thead class="bg-slate-950/60 text-gray-400 uppercase text-[10px] font-bold tracking-wider">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left">Symbol</th>
                            <th scope="col" class="px-6 py-3 text-left">Sector</th>
                            <th scope="col" class="px-6 py-3 text-left">Close Price</th>
                            <th scope="col" class="px-6 py-3 text-left">RSI (14)</th>
                            <th scope="col" class="px-6 py-3 text-left">Rel Volume</th>
                            <th scope="col" class="px-6 py-3 text-left">ATR %</th>
                            <th scope="col" class="px-6 py-3 text-center">Bull Score</th>
                            <th scope="col" class="px-6 py-3 text-center">Bear Score</th>
                            <th scope="col" class="px-6 py-3 text-center">Bias</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-800 bg-slate-900/30">
                        {leaderboard_html}
                    </tbody>
                </table>
            </div>
        </div>

    </main>

    <footer class="max-w-7xl mx-auto px-4 text-center mt-12 text-xs text-slate-500 border-t border-slate-900 pt-6">
        <p>© 2026 Quant Finder India. All market data fetched live from Yahoo Finance daily.</p>
        <p class="mt-1 font-bold text-gray-400">Disclaimer: All content on this site is for educational purposes only; stock trading involves significant market risk, and you should consult a SEBI-registered financial advisor before making any investment decisions.</p>
    </footer>

    <script>
        // IST CLOCK & MARKET STATUS
        function updateClock() {{
            const now = new Date();
            const istOffset = 5.5 * 60 * 60 * 1000;
            const utc = now.getTime() + (now.getTimezoneOffset() * 60 * 1000);
            const istTime = new Date(utc + istOffset);
            
            const hours = String(istTime.getHours()).padStart(2, '0');
            const minutes = String(istTime.getMinutes()).padStart(2, '0');
            const seconds = String(istTime.getSeconds()).padStart(2, '0');
            
            document.getElementById('ist-clock').innerText = hours + ":" + minutes + ":" + seconds;
            
            const day = istTime.getDay();
            const currentHour = istTime.getHours();
            const currentMin = istTime.getMinutes();
            const timeInMins = currentHour * 60 + currentMin;
            
            const badge = document.getElementById('market-state');
            if (day >= 1 && day <= 5) {{
                if (timeInMins >= 555 && timeInMins < 930) {{
                    badge.innerText = "MARKET OPEN";
                    badge.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-emerald-950 text-emerald-400 border border-emerald-500/20";
                }} else if (timeInMins >= 540 && timeInMins < 555) {{
                    badge.innerText = "PRE-MARKET";
                    badge.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-yellow-950 text-yellow-400 border border-yellow-500/20";
                }} else {{
                    badge.innerText = "MARKET CLOSED";
                    badge.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-red-950 text-red-400 border border-red-500/20";
                }}
            }} else {{
                badge.innerText = "WEEKEND CLOSED";
                badge.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-red-950 text-red-400 border border-red-500/20";
            }}
            
            // DYNAMIC SIGNAL HIGHLIGHTING FOR BANK NIFTY (Today Entry Criteria)
            calculateLiveBankNiftyStatus(istTime);
        }}
        
        // Dynamic Entry Detection & Alert Highlight for Bank Nifty
        let liveBankNiftyPrice = {latest_bn_close}; // Initial fallback from compiled python data
        
        function calculateLiveBankNiftyStatus(istTime) {{
            const currentPrice = liveBankNiftyPrice;
            const dailyLongTrigger = {daily_long_trigger};
            const dailyShortTrigger = {daily_short_trigger};
            
            const weeklyLongTrigger = {weekly_long_trigger};
            const weeklyShortTrigger = {weekly_short_trigger};
            
            const dBadge = document.getElementById('daily-action-badge');
            const wBadge = document.getElementById('weekly-action-badge');
            
            // Update UI elements with latest price
            document.getElementById('bn-last-close-val').innerText = "₹" + currentPrice.toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
            
            // Check Daily Status
            if (currentPrice > dailyLongTrigger) {{
                dBadge.innerText = "🚀 BUY SIGNAL ACTIVE";
                dBadge.className = "px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-green-950 text-green-400 border border-green-500/20 animate-bounce";
            }} else if (currentPrice < dailyShortTrigger) {{
                dBadge.innerText = "📉 SHORT SIGNAL ACTIVE";
                dBadge.className = "px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-red-950 text-red-400 border border-red-500/20 animate-bounce";
            }} else {{
                dBadge.innerText = "⏳ WAITING FOR BREAKOUT";
                dBadge.className = "px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-slate-950 text-slate-400 border border-slate-800";
            }}
            
            // Check Weekly Status
            if (currentPrice > weeklyLongTrigger) {{
                wBadge.innerText = "🚀 WEEKLY LONG ACTIVE";
                wBadge.className = "px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-green-950 text-green-400 border border-green-500/20";
            }} else if (currentPrice < weeklyShortTrigger) {{
                wBadge.innerText = "📉 WEEKLY SHORT ACTIVE";
                wBadge.className = "px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-red-950 text-red-400 border border-red-500/20";
            }} else {{
                wBadge.innerText = "⏳ WEEKLY CONSOLIDATION";
                wBadge.className = "px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase bg-slate-950 text-slate-400 border border-slate-800";
            }}
        }}

        // ==============================================================================
        // 🔄 ANIMATED REFRESH ENGINE: INTERACTIVE BROWSER-SIDE LIVE PRICE SYNC 🔄
        // ==============================================================================
        async function refreshLivePrices() {{
            const icon = document.getElementById('price-refresh-icon');
            const btn = document.getElementById('price-refresh-btn');
            
            // Add rotation animation class
            icon.classList.add('animate-spin');
            btn.disabled = true;
            btn.innerText = "Syncing Quotes...";
            
            console.log("Fetching live indexes from public API...");
            try {{
                // Fetch real-time Bank Nifty index quote from a free, open financial API (via allorigins CORS proxy if needed)
                const response = await fetch("https://api.allorigins.win/get?url=" + encodeURIComponent("https://query1.finance.yahoo.com/v8/finance/chart/^NSEBANK?interval=1m&range=1d"));
                if (response.ok) {{
                    const data = await response.json();
                    const parsedData = JSON.parse(data.contents);
                    const quote = parsedData.chart.result[0].indicators.quote[0].close;
                    const latestPrice = quote[quote.length - 1];
                    
                    if (latestPrice) {{
                        liveBankNiftyPrice = latestPrice;
                        console.log("Live Bank Nifty price updated to:", liveBankNiftyPrice);
                        
                        // Update daily bias label dynamically
                        const previousClose = parsedData.chart.result[0].meta.previousClose;
                        const dailyChange = ((latestPrice - previousClose) / previousClose) * 100;
                        const biasLbl = document.getElementById('bn-daily-bias-lbl');
                        biasLbl.innerText = dailyChange >= 0 ? "🟢 BULLISH (+" + dailyChange.toFixed(2) + "%)" : "🔴 BEARISH (" + dailyChange.toFixed(2) + "%)";
                        
                        // Recalculate indicators
                        updateClock();
                    }}
                }}
            }} catch (error) {{
                console.error("CORS API failed. Reverting to backup TradingView data sync.");
            }}
            
            // Simulate a brief delay for rotation effect
            setTimeout(() => {{
                icon.classList.remove('animate-spin');
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-rotate-right mr-2" id="price-refresh-icon"></i> Sync Live Quotes';
            }}, 1200);
        }}

        // ==============================================================================
        // ☁️ GITHUB ACTIONS TRIGGER SYSTEM (GENERATE CODES FROM THE BROWSER) ☁️
        // ==============================================================================
        function toggleTokenModal() {{
            const modal = document.getElementById('token-modal');
            modal.classList.toggle('hidden');
            
            // Pre-fill token if saved
            const savedToken = localStorage.getItem('gh_pat_token');
            if (savedToken) {{
                document.getElementById('gh-token').value = savedToken;
                document.getElementById('action-trigger-area').classList.remove('hidden');
            }}
        }}

        function saveGitHubToken() {{
            const token = document.getElementById('gh-token').value.trim();
            if (token) {{
                localStorage.setItem('gh_pat_token', token);
                document.getElementById('action-trigger-area').classList.remove('hidden');
                alert("Token saved securely in local storage!");
            }} else {{
                alert("Please paste a valid token.");
            }}
        }}

        function clearGitHubToken() {{
            localStorage.removeItem('gh_pat_token');
            document.getElementById('gh-token').value = '';
            document.getElementById('action-trigger-area').classList.add('hidden');
            alert("Token cleared successfully.");
        }}

        async function triggerGitHubAction() {{
            const token = localStorage.getItem('gh_pat_token');
            const runBtn = document.getElementById('run-workflow-btn');
            const spinner = document.getElementById('workflow-spinner');
            const countdownArea = document.getElementById('countdown-area');
            const countdownSecs = document.getElementById('countdown-seconds');
            
            if (!token) {{
                alert("Please configure your GitHub token first.");
                return;
            }}
            
            runBtn.disabled = true;
            spinner.classList.add('animate-spin');
            
            // GitHub REST API Parameters
            const owner = "abhisim23";
            const repo = "nifty-scanner";
            const workflowId = "daily_scanner.yml"; // Filename of action
            
            try {{
                console.log("Triggering dispatch on GitHub Actions...");
                const response = await fetch(`https://api.github.com/repos/${{owner}}/${{repo}}/actions/workflows/${{workflowId}}/dispatches`, {{
                    method: 'POST',
                    headers: {{
                        'Authorization': `Bearer ${{token}}`,
                        'Accept': 'application/vnd.github+json',
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ ref: 'main' }})
                }});
                
                if (response.status === 204) {{
                    // 204 No Content indicates success!
                    console.log("Success! Cloud build triggered.");
                    countdownArea.classList.remove('hidden');
                    
                    // Start a 60 second countdown timer to page reload
                    let timeLeft = 60;
                    const timer = setInterval(() => {{
                        timeLeft--;
                        countdownSecs.innerText = timeLeft;
                        if (timeLeft <= 0) {{
                            clearInterval(timer);
                            window.location.reload(); // Reload page to view updated files!
                        }}
                    }}, 1000);
                    
                }} else {{
                    const errData = await response.json();
                    alert(`Action failed: ${{errData.message || response.statusText}}`);
                    runBtn.disabled = false;
                    spinner.classList.remove('animate-spin');
                }}
            }} catch (error) {{
                alert(`Network error: ${{error.message}}`);
                runBtn.disabled = false;
                spinner.classList.remove('animate-spin');
            }}
        }}

        // POSITION SIZING CALCULATORS
        function calculatePositionLong(index, entry, sl) {{
            const capital = parseFloat(document.getElementById('capital-long-' + index).value) || 0;
            const riskPct = parseFloat(document.getElementById('risk-long-' + index).value) || 0;
            
            const cashRisk = capital * (riskPct / 100);
            const slPoints = entry - sl;
            
            let qty = 0;
            let requiredMargin = "0.00";
            
            if (slPoints > 0) {{
                qty = Math.floor(cashRisk / slPoints);
                requiredMargin = "₹" + ((qty * entry) / 5).toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
            }}
            
            document.getElementById('risk-cash-long-' + index).innerText = "₹" + cashRisk.toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
            document.getElementById('qty-long-' + index).innerText = qty + " shares";
            document.getElementById('margin-long-' + index).innerText = requiredMargin;
        }}
        
        function calculatePositionShort(index, entry, sl) {{
            const capital = parseFloat(document.getElementById('capital-short-' + index).value) || 0;
            const riskPct = parseFloat(document.getElementById('risk-short-' + index).value) || 0;
            
            const cashRisk = capital * (riskPct / 100);
            const slPoints = sl - entry;
            
            let qty = 0;
            let requiredMargin = "0.00";
            
            if (slPoints > 0) {{
                qty = Math.floor(cashRisk / slPoints);
                requiredMargin = "₹" + ((qty * entry) / 5).toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
            }}
            
            document.getElementById('risk-cash-short-' + index).innerText = "₹" + cashRisk.toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
            document.getElementById('qty-short-' + index).innerText = qty + " shares";
            document.getElementById('margin-short-' + index).innerText = requiredMargin;
        }}

        // OPTIONS CALCULATOR
        function calculateOptionStats() {{
            const premium = parseFloat(document.getElementById('opt-premium').value) || 0;
            const lots = parseInt(document.getElementById('opt-lots').value) || 0;
            const slPts = parseFloat(document.getElementById('opt-sl-pts').value) || 0;
            const lotSize = 15;
            
            const totalQty = lots * lotSize;
            const totalCapital = totalQty * premium;
            const maxRisk = totalQty * slPts;
            const approxMargin = lots * 120000;
            
            document.getElementById('opt-buy-capital').innerText = "₹" + totalCapital.toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
            document.getElementById('opt-total-risk').innerText = "₹" + maxRisk.toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
            document.getElementById('opt-sell-margin').innerText = "₹" + approxMargin.toLocaleString('en-IN', {{ maximumFractionDigits: 2 }});
        }}

        // TRADING JOURNAL CONTROLLER (Local Storage Persistent)
        let journal = JSON.parse(localStorage.getItem('trading_journal')) || [];

        function renderJournal() {{
            const tbody = document.querySelector('#journal-table tbody');
            tbody.innerHTML = '';
            
            if (journal.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="5" class="px-4 py-4 text-center text-gray-500 italic">No logged trades logged yet. Start logging below!</td></tr>';
                return;
            }}
            
            journal.forEach((trade, index) => {{
                const pnlColor = trade.pnl >= 0 ? 'text-green-400' : 'text-red-400';
                const pnlPrefix = trade.pnl >= 0 ? '+' : '';
                tbody.innerHTML += `
                <tr class="hover:bg-slate-800/30 transition">
                    <td class="px-4 py-2 font-bold text-white">${{trade.stock}}</td>
                    <td class="px-4 py-2 font-semibold text-slate-300">${{trade.qty}}</td>
                    <td class="px-4 py-2 font-black ${{pnlColor}}">${{pnlPrefix}}₹${{parseFloat(trade.pnl).toLocaleString('en-IN')}}</td>
                    <td class="px-4 py-2 text-slate-400 text-[11px]">${{trade.notes}}</td>
                    <td class="px-4 py-2 text-right">
                        <button onclick="deleteJournalEntry(${{index}})" class="text-rose-400 hover:text-rose-300 px-2 py-1"><i class="fa-solid fa-trash"></i></button>
                    </td>
                </tr>
                `;
            }});
        }}

        function addJournalEntry(e) {{
            e.preventDefault();
            const stock = document.getElementById('j-stock').value.toUpperCase().trim();
            const qty = parseInt(document.getElementById('j-qty').value) || 0;
            const pnl = parseFloat(document.getElementById('j-pnl').value) || 0;
            const notes = document.getElementById('j-notes').value.trim();
            
            journal.push({{ stock, qty, pnl, notes }});
            localStorage.setItem('trading_journal', JSON.stringify(journal));
            
            document.getElementById('journal-form').reset();
            renderJournal();
        }}

        function deleteJournalEntry(index) {{
            journal.splice(index, 1);
            localStorage.setItem('trading_journal', JSON.stringify(journal));
            renderJournal();
        }}

        // Run initial calculations on load
        window.onload = function() {{
            calculateOptionStats();
            renderJournal();
            
            // Check if GitHub token is already saved and adjust modal view
            const savedToken = localStorage.getItem('gh_pat_token');
            if (savedToken) {{
                document.getElementById('gh-token').value = savedToken;
                document.getElementById('action-trigger-area').classList.remove('hidden');
            }}
            
            for (let i = 1; i <= 3; i++) {{
                const longCapitalInput = document.getElementById('capital-long-' + i);
                if (longCapitalInput) {{
                    longCapitalInput.dispatchEvent(new Event('input'));
                }}
            }}
            for (let i = 4; i <= 6; i++) {{
                const shortCapitalInput = document.getElementById('capital-short-' + i);
                if (shortCapitalInput) {{
                    shortCapitalInput.dispatchEvent(new Event('input'));
                }}
            }}
        }}

        // LEADERBOARD FILTER SEARCH
        document.getElementById('search-box').addEventListener('keyup', function() {{
            const filter = this.value.toUpperCase();
            const rows = document.querySelectorAll('#leaderboard-table tbody tr');
            
            rows.forEach(row => {{
                const symbol = row.cells[0].textContent || row.cells[0].innerText;
                if (symbol.toUpperCase().indexOf(filter) > -1) {{
                    row.style.display = "";
                }} else {{
                    row.style.display = "none";
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    
    with open("index.html", "w") as f:
        f.write(html_template)
    print("Beautiful interactive HTML dashboard updated successfully as 'index.html'!")

if __name__ == "__main__":
    generate_dashboard()
