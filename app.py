import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="TQQQ 퀀트 백테스터", layout="wide")
st.title("🦅 TQQQ 실전 퀀트 전략 백테스터")

# --- 사이드바 설정 (사용자 입력) ---
st.sidebar.header("⚙️ 전략 및 기간 설정")
strategy_choice = st.sidebar.radio("💡 백테스트할 전략을 선택하세요", ["1. TQQQ RSI 40일 스윙", "2. 라오어 무한매수법"])

st.sidebar.markdown("---")
BACKTEST_START = st.sidebar.date_input("백테스트 시작일", datetime(2022, 1, 1))
BACKTEST_END = st.sidebar.date_input("백테스트 종료일", datetime.today())

st.sidebar.markdown("---")
if strategy_choice == "1. TQQQ RSI 40일 스윙":
    st.sidebar.subheader("🔥 RSI 스윙 설정")
    INITIAL_CAPITAL = st.sidebar.number_input("초기 투자 자본금 ($)", value=10000.0, step=1000.0, format="%.2f")
    MULTIPLIER = st.sidebar.number_input("매수 수량 배수 (기본 1배)", min_value=1, max_value=20, value=1, step=1)
    QQQ_MA_FILTER = st.sidebar.selectbox("🛡️ 하락장 매수 중단 (QQQ 이평선)", ["사용 안 함", "120일선", "150일선", "200일선"])
    
    st.markdown(f"""
    ### 📊 전략 1. TQQQ RSI 40일 스윙 (현재 설정: {MULTIPLIER}배수)
    * **방어 룰:** QQQ가 **{QQQ_MA_FILTER}** 아래로 내려가면 신규 매수를 전면 중단합니다.
    * **매수:** RSI < 40 (**{4 * MULTIPLIER}주**) / 40~50 (**{3 * MULTIPLIER}주**) / 50~60 (**{2 * MULTIPLIER}주**) / 60~70 (**{1 * MULTIPLIER}주**) / 70 이상 (매수 안함)
    * **매도 (익절):** 평균 단가 대비 **+7.5%** 수익 도달 시 전량 매도
    * **타임아웃 (40일 경과 시):** 1. 수익률 **-10% 이상 ~ +7.5% 미만** ➔ 즉시 전량 매도 (손익 마감)
        2. 수익률 **-10% 미만** ➔ 매수 중단 후 홀딩, **본전(0%)** 회복 시 전량 매도
    """)
else:
    st.sidebar.subheader("🔥 라오어 무한매수법 설정")
    INITIAL_CAPITAL = st.sidebar.number_input("초기 투자 자본금 ($)", value=10000.0, step=1000.0, format="%.2f")
    SPLITS = st.sidebar.number_input("분할 횟수 (기본 40일)", min_value=10, max_value=100, value=40, step=1)
    TARGET_RETURN = st.sidebar.number_input("기대 수익률 (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
    # 신규 추가: 라오어 전략용 이평선 필터
    QQQ_MA_FILTER = st.sidebar.selectbox("🛡️ 하락장 매수 중단 (QQQ 이평선)", ["사용 안 함", "120일선", "150일선", "200일선"], key="laore_ma")
    
    st.markdown(f"""
    ### 📊 전략 2. 라오어 무한매수법 (현재 설정: {SPLITS}분할, 목표 +{TARGET_RETURN}%)
    * **방어 룰:** QQQ가 **{QQQ_MA_FILTER}** 아래로 내려가면 기계적 매수를 일시 중단하고 현금을 아낍니다. (보유 기간 카운트는 유지됨)
    * **매수:** 초기 자본금(${INITIAL_CAPITAL:,.0f})을 {SPLITS}등분 하여, 매일 **${INITIAL_CAPITAL/SPLITS:,.2f}** 씩 TQQQ를 기계적으로 매수합니다.
    * **매도 (익절):** 평균 단가 대비 **+{TARGET_RETURN}%** 수익 도달 시 전량 매도 후 다음 날부터 새 사이클을 시작합니다.
    * **원금 소진 / 타임아웃:** {SPLITS}일이 경과하여 사이클이 종료되었음에도 목표 수익률에 도달하지 못하면, **즉시 전량 매도(손절)**하고 새로운 사이클로 리셋합니다.
    """)

# --- 데이터 수집 ---
@st.cache_data(ttl=3600)
def fetch_data(start_date, end_date):
    fetch_start = start_date - timedelta(days=300) # MA200 계산을 위해 넉넉히 수집
    data = yf.download(['TQQQ', 'QQQ'], start=fetch_start.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)['Close']
    df = pd.DataFrame(index=data.index)
    df['TQQQ'] = data['TQQQ'].ffill()
    df['QQQ'] = data['QQQ'].ffill()
    
    df['QQQ_MA120'] = df['QQQ'].rolling(120).mean()
    df['QQQ_MA150'] = df['QQQ'].rolling(150).mean()
    df['QQQ_MA200'] = df['QQQ'].rolling(200).mean()
    df['RSI'] = ta.rsi(df['TQQQ'], length=14)
    
    return df.dropna().loc[pd.to_datetime(start_date):]

# --- 1. RSI 스윙 백테스트 엔진 ---
def run_rsi_backtest(df, initial_cash, multiplier, qqq_ma_filter):
    cash = initial_cash
    cycles = []
    equity_curve = []
    
    current_cycle = {'start_date': None, 'invested': 0.0, 'shares': 0, 'days': 0, 'peak_val': 0.0, 'mdd': 0.0, 'status': 'buying'}
    avg_price = 0.0
    
    ma_col = None
    if qqq_ma_filter == "120일선": ma_col = 'QQQ_MA120'
    elif qqq_ma_filter == "150일선": ma_col = 'QQQ_MA150'
    elif qqq_ma_filter == "200일선": ma_col = 'QQQ_MA200'
    
    for date, row in df.iterrows():
        price = row['TQQQ']
        qqq_price = row['QQQ']
        rsi = row['RSI']
        port_val = cash + (current_cycle['shares'] * price)
        
        # 보유 중 매도 체크
        if current_cycle['shares'] > 0:
            ret = (price - avg_price) / avg_price
            current_cycle['days'] += 1
            
            current_cycle['peak_val'] = max(current_cycle['peak_val'], port_val)
            dd = (port_val / current_cycle['peak_val']) - 1
            current_cycle['mdd'] = min(current_cycle['mdd'], dd)
            
            sell, sell_reason = False, ""
            
            if current_cycle['status'] == 'buying':
                if ret >= 0.075:
                    sell, sell_reason = True, "목표 수익(+7.5%) 달성 익절"
                elif current_cycle['days'] >= 40:
                    if ret >= -0.10:
                        sell, sell_reason = True, "40일 타임아웃 (-10% 이상 강제 매도)"
                    else:
                        current_cycle['status'] = 'waiting_breakeven' 
            elif current_cycle['status'] == 'waiting_breakeven':
                if ret >= 0.0:
                    sell, sell_reason = True, "존버 후 본전(0%) 도달 탈출"
                    
            if sell:
                cash += current_cycle['shares'] * price
                profit = (current_cycle['shares'] * price) - current_cycle['invested']
                profit_pct = profit / current_cycle['invested']
                
                cycles.append({
                    '사이클 시작일': current_cycle['start_date'].strftime('%Y-%m-%d'),
                    '사이클 종료일': date.strftime('%Y-%m-%d'),
                    '투자 기간 (일)': current_cycle['days'],
                    '투입 금액 ($)': current_cycle['invested'],
                    '수익 금액 ($)': profit,
                    '수익률 (%)': profit_pct * 100,
                    '사이클 MDD (%)': current_cycle['mdd'] * 100,
                    '매도 사유': sell_reason
                })
                
                current_cycle = {'start_date': None, 'invested': 0.0, 'shares': 0, 'days': 0, 'peak_val': 0.0, 'mdd': 0.0, 'status': 'buying'}
                avg_price = 0.0
                port_val = cash 
                
        # 매수 로직
        can_buy = True
        if ma_col and pd.notna(row[ma_col]):
            if qqq_price < row[ma_col]:
                can_buy = False # QQQ가 이평선 아래면 매수 금지
                
        if current_cycle['status'] == 'buying' and can_buy:
            base_shares = 0
            if pd.notna(rsi):
                if rsi < 40: base_shares = 4
                elif 40 <= rsi < 50: base_shares = 3
                elif 50 <= rsi < 60: base_shares = 2
                elif 60 <= rsi < 70: base_shares = 1
                
            shares_to_buy = base_shares * multiplier
                
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                if cash < cost:
                    shares_to_buy = int(cash // price)
                    cost = shares_to_buy * price
                    
                if shares_to_buy > 0:
                    if current_cycle['shares'] == 0:
                        current_cycle['start_date'] = date
                        current_cycle['peak_val'] = port_val
                    total_cost_prev = current_cycle['shares'] * avg_price
                    current_cycle['shares'] += shares_to_buy
                    current_cycle['invested'] += cost
                    avg_price = (total_cost_prev + cost) / current_cycle['shares']
                    cash -= cost
                    port_val = cash + (current_cycle['shares'] * price)
                    
        equity_curve.append({'Date': date, 'Total Equity': port_val, 'Cash': cash, 'Invested': current_cycle['invested']})
        
    return pd.DataFrame(equity_curve).set_index('Date'), pd.DataFrame(cycles), current_cycle, avg_price

# --- 2. 라오어 무한매수법 백테스트 엔진 ---
def run_laore_backtest(df, initial_cash, splits, target_return_pct, qqq_ma_filter):
    daily_buy_amt = initial_cash / splits
    target_return = target_return_pct / 100.0
    
    cash = initial_cash
    cycles = []
    equity_curve = []
    
    current_cycle = {'start_date': None, 'invested': 0.0, 'shares': 0, 'days': 0, 'peak_val': 0.0, 'mdd': 0.0, 'status': 'buying'}
    avg_price = 0.0
    
    # 이평선 필터 컬럼 매핑
    ma_col = None
    if qqq_ma_filter == "120일선": ma_col = 'QQQ_MA120'
    elif qqq_ma_filter == "150일선": ma_col = 'QQQ_MA150'
    elif qqq_ma_filter == "200일선": ma_col = 'QQQ_MA200'
    
    for date, row in df.iterrows():
        price = row['TQQQ']
        qqq_price = row['QQQ']
        port_val = cash + (current_cycle['shares'] * price)
        
        # 매도 체크 (보유 중일 때)
        if current_cycle['shares'] > 0:
            ret = (price - avg_price) / avg_price
            current_cycle['days'] += 1 # 매수를 쉬더라도 보유 일수는 흘러감
            
            current_cycle['peak_val'] = max(current_cycle['peak_val'], port_val)
            dd = (port_val / current_cycle['peak_val']) - 1
            current_cycle['mdd'] = min(current_cycle['mdd'], dd)
            
            sell, sell_reason = False, ""
            
            if ret >= target_return:
                sell, sell_reason = True, f"목표 수익(+{target_return_pct}%) 달성 익절"
            elif current_cycle['days'] >= splits:
                sell, sell_reason = True, f"시간 종료({splits}일 타임아웃) 강제 청산"
                
            if sell:
                cash += current_cycle['shares'] * price
                profit = (current_cycle['shares'] * price) - current_cycle['invested']
                profit_pct = profit / current_cycle['invested']
                
                cycles.append({
                    '사이클 시작일': current_cycle['start_date'].strftime('%Y-%m-%d'),
                    '사이클 종료일': date.strftime('%Y-%m-%d'),
                    '투자 기간 (일)': current_cycle['days'],
                    '투입 금액 ($)': current_cycle['invested'],
                    '수익 금액 ($)': profit,
                    '수익률 (%)': profit_pct * 100,
                    '사이클 MDD (%)': current_cycle['mdd'] * 100,
                    '매도 사유': sell_reason
                })
                
                current_cycle = {'start_date': None, 'invested': 0.0, 'shares': 0, 'days': 0, 'peak_val': 0.0, 'mdd': 0.0, 'status': 'buying'}
                avg_price = 0.0
                port_val = cash # 오늘 팔았으니 내일부터 매수 시작
                
        # 매수 로직 (하락장 방어 필터 적용)
        can_buy = True
        if ma_col and pd.notna(row[ma_col]):
            if qqq_price < row[ma_col]:
                can_buy = False # QQQ가 이평선 아래면 무지성 매수 일시 정지
                
        if current_cycle['status'] == 'buying' and current_cycle['days'] < splits and can_buy:
            if current_cycle['shares'] == 0 or (current_cycle['shares'] > 0 and current_cycle['start_date'] != date):
                cost_to_spend = min(daily_buy_amt, cash)
                shares_to_buy = int(cost_to_spend // price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    if current_cycle['shares'] == 0:
                        current_cycle['start_date'] = date
                        current_cycle['peak_val'] = port_val
                        
                    total_cost_prev = current_cycle['shares'] * avg_price
                    current_cycle['shares'] += shares_to_buy
                    current_cycle['invested'] += cost
                    avg_price = (total_cost_prev + cost) / current_cycle['shares']
                    cash -= cost
                    port_val = cash + (current_cycle['shares'] * price)
                
        equity_curve.append({'Date': date, 'Total Equity': port_val, 'Cash': cash, 'Invested': current_cycle['invested']})
        
    return pd.DataFrame(equity_curve).set_index('Date'), pd.DataFrame(cycles), current_cycle, avg_price

# --- 백테스트 실행부 ---
with st.spinner("알고리즘이 백테스트를 수행 중입니다..."):
    df_raw = fetch_data(BACKTEST_START, BACKTEST_END)
    
    if strategy_choice == "1. TQQQ RSI 40일 스윙":
        eq_df, cyc_df, cur_cyc, avg_price = run_rsi_backtest(df_raw, INITIAL_CAPITAL, MULTIPLIER, QQQ_MA_FILTER)
    else:
        eq_df, cyc_df, cur_cyc, avg_price = run_laore_backtest(df_raw, INITIAL_CAPITAL, SPLITS, TARGET_RETURN, QQQ_MA_FILTER)

# --- 결과 요약 대시보드 ---
st.subheader("📊 백테스트 종합 결과")

final_equity = eq_df['Total Equity'].iloc[-1]
total_return = (final_equity / INITIAL_CAPITAL) - 1
total_days = (eq_df.index[-1] - eq_df.index[0]).days
max_dd = (eq_df['Total Equity'] / eq_df['Total Equity'].cummax() - 1).min()

win_rate = 0.0
total_trades = len(cyc_df)
if total_trades > 0:
    win_rate = (len(cyc_df[cyc_df['수익 금액 ($)'] > 0]) / total_trades) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("최종 자산 ($)", f"${final_equity:,.2f}")
col2.metric("총 수익률 (%)", f"{total_return*100:+.2f}%")
col3.metric("계좌 최대 낙폭 (MDD)", f"{max_dd*100:.2f}%")
col4.metric("완료된 사이클 횟수", f"{total_trades} 회")
col5.metric("사이클 승률", f"{win_rate:.1f}%")

st.divider()

# --- 현재 진행 중인 사이클 현황 ---
st.subheader("📡 현재 진행 중인 사이클 현황")
if cur_cyc['shares'] > 0:
    curr_price = df_raw['TQQQ'].iloc[-1]
    curr_ret = (curr_price - avg_price) / avg_price
    
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("진행 기간 (일)", f"{cur_cyc['days']} 일")
    s_col2.metric("현재 투입 금액", f"${cur_cyc['invested']:,.2f}")
    s_col3.metric("보유 수량 및 평단가", f"{cur_cyc['shares']}주 / ${avg_price:.2f}")
    
    ret_color = "normal" if curr_ret >= 0 else "inverse"
    s_col4.metric("현재 수익률", f"{curr_ret*100:+.2f}%", delta_color=ret_color)
    
    if cur_cyc['status'] == 'buying':
        st.info("🟢 **상태:** 사이클 매수 진행 중 (목표 수익 도달 또는 타임아웃 대기 중)")
    else:
        st.error("🔴 **상태:** 타임아웃! 강제 존버 모드 (추가 매수 없이 본전 회복 대기 중)")
else:
    st.success("✅ 현재 모든 사이클이 익절/청산되어 **현금 100%** 보유 중입니다. 다음 매수 타점을 대기합니다.")

st.divider()

# --- 사이클별 상세 결과 (표) ---
st.subheader("📋 사이클별 매매 상세 내역")
if not cyc_df.empty:
    styled_cyc = cyc_df.copy()
    styled_cyc['투입 금액 ($)'] = styled_cyc['투입 금액 ($)'].apply(lambda x: f"${x:,.2f}")
    styled_cyc['수익 금액 ($)'] = styled_cyc['수익 금액 ($)'].apply(lambda x: f"${x:+,.2f}")
    
    def color_returns(val):
        if type(val) == float or type(val) == int:
            if val > 0: return 'color: #2ecc71; font-weight: bold;'
            elif val < 0: return 'color: #e74c3c; font-weight: bold;'
        return ''

    st.dataframe(styled_cyc.style.map(color_returns, subset=['수익률 (%)', '사이클 MDD (%)']), hide_index=True, use_container_width=True)
else:
    st.info("아직 완료된 사이클이 없습니다.")

st.divider()

# --- 자산 성장 곡선 ---
st.subheader("📈 전체 자산 성장 곡선 및 투입 원금 변화")
fig = go.Figure()
fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df['Total Equity'], mode='lines', name='총 자산 (현금+주식)', line=dict(color='#8e44ad', width=2.5)))
fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df['Invested'], mode='lines', name='주식 투입 금액', fill='tozeroy', line=dict(color='#e74c3c', width=1.5, dash='dot')))

fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="달러 ($)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)
