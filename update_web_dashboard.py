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
    print("Fetching market data...")
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
                
            # Date detection
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            last_row_date = df.index[-1].strftime('%Y-%m-%d')
            
            # Use yesterday if today's candle is active and incomplete
            if last_row_date == today_str:
                analysis_idx = -2
                target_date = df.index[-2]
                trading_date = df.index[-1]
            else:
                analysis_idx = -1
                target_date = df.index[-1]
                trading_date = datetime.date.today()
                
            # Compute technical indicators
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
            
            # Squeeze Percentile
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
            
            results.append({
                'ticker': ticker,
                'symbol': ticker.replace(".NS", ""),
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
        
    longs = df_all[df_all['bull_score'] >= 45].sort_values(by='bull_score', ascending=False).head(3)
    shorts = df_all[df_all['bear_score'] >= 45].sort_values(by='bear_score', ascending=False).head(3)
    
    target_date_str = longs['target_date'].iloc[0] if not longs.empty else df_all['target_date'].iloc[0]
    trading_date_str = longs['trading_date'].iloc[0] if not longs.empty else df_all['trading_date'].iloc[0]
    
    # Generate HTML Cards
    longs_html = ""
    for idx, row in longs.iterrows():
        entry = max(row['close'] * 1.002, row['high'])
        sl = entry - (row['atr'] * 0.5)
        t1 = entry + (row['atr'] * 0.75)
        t2 = entry + (row['atr'] * 1.5)
        longs_html += f"""
        <div class="bg-gray-800 border border-green-500 rounded-xl p-6 shadow-lg hover:shadow-2xl transition duration-300">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <span class="bg-green-900 text-green-300 text-xs font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">BUY LONG</span>
                    <h3 class="text-2xl font-black text-white mt-2">{row['symbol']}</h3>
                    <p class="text-xs text-gray-400 font-medium">{row['ticker']}</p>
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
            
            <div class="space-y-2 mb-6">
                <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                    <span class="text-green-400 font-bold uppercase">Trigger Entry (Above)</span>
                    <span class="font-black text-white">₹{entry:,.2f}</span>
                </div>
                <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                    <span class="text-red-400 font-bold uppercase">Stop-Loss (SL)</span>
                    <span class="font-black text-white">₹{sl:,.2f}</span>
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
        
    shorts_html = ""
    for idx, row in shorts.iterrows():
        entry = min(row['close'] * 0.998, row['low'])
        sl = entry + (row['atr'] * 0.5)
        t1 = entry - (row['atr'] * 0.75)
        t2 = entry - (row['atr'] * 1.5)
        shorts_html += f"""
        <div class="bg-gray-800 border border-red-500 rounded-xl p-6 shadow-lg hover:shadow-2xl transition duration-300">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <span class="bg-red-900 text-red-300 text-xs font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">SHORT SELL</span>
                    <h3 class="text-2xl font-black text-white mt-2">{row['symbol']}</h3>
                    <p class="text-xs text-gray-400 font-medium">{row['ticker']}</p>
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
            
            <div class="space-y-2 mb-6">
                <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                    <span class="text-red-400 font-bold uppercase">Trigger Entry (Below)</span>
                    <span class="font-black text-white">₹{entry:,.2f}</span>
                </div>
                <div class="bg-gray-900 p-2.5 rounded border border-gray-700 flex justify-between text-sm">
                    <span class="text-green-400 font-bold uppercase">Stop-Loss (SL)</span>
                    <span class="font-black text-white">₹{sl:,.2f}</span>
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
        
        <!-- INTRO HERO BANNER -->
        <div class="bg-gradient-to-r from-slate-900 to-indigo-950 rounded-2xl p-6 sm:p-8 border border-slate-800 mb-8 shadow-xl relative overflow-hidden">
            <div class="absolute -right-12 -bottom-12 opacity-10 text-9xl font-black">NSE</div>
            <div class="max-w-3xl relative z-10">
                <h2 class="text-2xl sm:text-3xl font-black text-white mb-2">High-Probability Intraday Stock Scanner</h2>
                <p class="text-slate-400 text-sm sm:text-base leading-relaxed mb-4">
                    Quantitatively scanning the top 50 most liquid NSE stocks every evening. This algorithmic model filters for institutional volume spikes, relative range expansions (ATR), trend alignment, and breakout squeezes to output the 3 strongest setups for the day.
                </p>
                <div class="flex flex-wrap gap-4 text-xs font-semibold text-slate-300">
                    <span class="flex items-center gap-1.5"><i class="fa-solid fa-shield-halved text-emerald-400"></i> Risk-to-Reward Optimized</span>
                    <span class="flex items-center gap-1.5"><i class="fa-solid fa-bolt text-indigo-400"></i> 15-Min ORB Trigger</span>
                    <span class="flex items-center gap-1.5"><i class="fa-solid fa-code text-pink-400"></i> Fully Automated</span>
                </div>
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
        <p class="mt-1">Designed with professional risk-to-reward parameters for Indian retail traders.</p>
    </footer>

    <script>
        // IST CLOCK & MARKET STATUS
        function updateClock() {{
            const now = new Date();
            // Convert to IST timezone
            const istOffset = 5.5 * 60 * 60 * 1000;
            const utc = now.getTime() + (now.getTimezoneOffset() * 60 * 1000);
            const istTime = new Date(utc + istOffset);
            
            const hours = String(istTime.getHours()).padStart(2, '0');
            const minutes = String(istTime.getMinutes()).padStart(2, '0');
            const seconds = String(istTime.getSeconds()).padStart(2, '0');
            
            document.getElementById('ist-clock').innerText = hours + ":" + minutes + ":" + seconds;
            
            // Check Indian Stock Market Status (09:15 to 15:30, Monday to Friday)
            const day = istTime.getDay(); // 0 is Sunday, 6 is Saturday
            const currentHour = istTime.getHours();
            const currentMin = istTime.getMinutes();
            const timeInMins = currentHour * 60 + currentMin;
            
            const badge = document.getElementById('market-state');
            if (day >= 1 && day <= 5) {{
                if (timeInMins >= 555 && timeInMins < 930) {{ // 09:15 AM to 03:30 PM
                    badge.innerText = "MARKET OPEN";
                    badge.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-emerald-950 text-emerald-400 border border-emerald-500/20";
                }} else if (timeInMins >= 540 && timeInMins < 555) {{ // 09:00 AM to 09:15 AM
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
        }}
        
        setInterval(updateClock, 1000);
        updateClock();

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
