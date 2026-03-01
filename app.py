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
st.set_page_config(page_title="TQQQ RSI 스윙 퀀트 대시보드", layout="wide")
st.title("🦅 TQQQ RSI(14) 40일 스윙 전략 백테스터")
st.markdown("""
**[전략 룰]** * **매수:** RSI < 40 (4주) / 40~50 (3주) / 50~60 (2주) / 60~70 (1주) / 70 이상 (매수 안함)
* **매도 (익절):** 평균 단가 대비 **+7.5%** 수익 도달 시 전량 매도
* **타임아웃 (40일 경과 시):** 1. 수익률 **-10% 이상 ~ +7.5% 미만** ➔ 즉시 전량 매도 (약손실 마감)
    2. 수익률 **-10% 미만** ➔ 매수 중단 후 홀딩, **본전(0%)** 회복 시 전량 매도
""")

# --- 사이드바 설정 (사용자 입력) ---
st.sidebar.header("⚙️ 백테스트 설정")
BACKTEST_START = st.sidebar.date_input("백테스트 시작일", datetime(2022, 1, 1))
BACKTEST_END = st.sidebar.date_input("백테스트 종료일", datetime.today())
INITIAL_CAPITAL = st.sidebar.number_input("초기 투자 자본금 ($)", value=10000.0, step=1000.0, format="%.2f")

# --- 데이터 수집 및 백테스트 엔진 ---
@st.cache_data(ttl=3600)
def run_backtest(start_date, end_date, initial_cash):
    # RSI 계산을 위해 시작일보다 40일 전 데이터부터 수집
    fetch_start = start_date - timedelta(days=60)
    data = yf.download("TQQQ", start=fetch_start.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)['Close']
    
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
        
    df = pd.DataFrame(index=data.index)
    df['Close'] = data
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df = df.dropna()
    
    # 지정한 시작일부터 백테스트 진행
    df = df.loc[pd.to_datetime(start_date):]
    
    cash = initial_cash
    cycles = []
    equity_curve = []
    
    # 사이클 변수 초기화
    current_cycle = {
        'start_date': None,
        'invested': 0.0,
        'shares': 0,
        'days': 0,
        'peak_val': 0.0,
        'mdd': 0.0,
        'status': 'buying' # 'buying' 또는 'waiting_breakeven'
    }
    avg_price = 0.0
    
    for date, row in df.iterrows():
        price = row['Close']
        rsi = row['RSI']
        
        # 1. 포트폴리오 가치 업데이트
        port_val = cash + (current_cycle['shares'] * price)
        
        # 2. 보유 중일 경우 상태 및 매도 조건 체크
        if current_cycle['shares'] > 0:
            ret = (price - avg_price) / avg_price
            current_cycle['days'] += 1
            
            # 사이클 내 MDD 계산 (최고점 갱신 및 낙폭 기록)
            current_cycle['peak_val'] = max(current_cycle['peak_val'], port_val)
            dd = (port_val / current_cycle['peak_val']) - 1
            current_cycle['mdd'] = min(current_cycle['mdd'], dd)
            
            sell = False
            sell_reason = ""
            
            if current_cycle['status'] == 'buying':
                if ret >= 0.075:
                    sell = True
                    sell_reason = "목표 수익(+7.5%) 달성 익절"
                elif current_cycle['days'] >= 40:
                    if ret >= -0.10:
                        sell = True
                        sell_reason = "40일 타임아웃 (-10% 이상 강제 매도)"
                    else:
                        current_cycle['status'] = 'waiting_breakeven' # 매수 중단 및 존버 돌입
                        
            elif current_cycle['status'] == 'waiting_breakeven':
                if ret >= 0.0:
                    sell = True
                    sell_reason = "존버 후 본전(0%) 도달 탈출"
                    
            # 매도 실행
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
                
                # 사이클 리셋
                current_cycle = {
                    'start_date': None, 'invested': 0.0, 'shares': 0, 'days': 0,
                    'peak_val': 0.0, 'mdd': 0.0, 'status': 'buying'
                }
                avg_price = 0.0
                port_val = cash # 매도 당일 종가 기준 포트폴리오 가치
                
        # 3. 매수 로직 (buying 상태이고, 아직 40일이 안 지났을 때)
        if current_cycle['status'] == 'buying':
            shares_to_buy = 0
            if pd.notna(rsi):
                if rsi < 40: shares_to_buy = 4
                elif 40 <= rsi < 50: shares_to_buy = 3
                elif 50 <= rsi < 60: shares_to_buy = 2
                elif 60 <= rsi < 70: shares_to_buy = 1
                
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                # 현금이 부족하면 살 수 있는 만큼만 삼
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
                    port_val = cash + (current_cycle['shares'] * price) # 매수 반영 후 가치
                    
        # 4. 일일 자산 기록
        equity_curve.append({
            'Date': date,
            'Total Equity': port_val,
            'Cash': cash,
            'Invested': current_cycle['invested'],
            'RSI': rsi
        })
        
    eq_df = pd.DataFrame(equity_curve).set_index('Date')
    cyc_df = pd.DataFrame(cycles)
    
    # 마지막 안 끝난 사이클이 있으면 상태 텍스트화
    current_status = None
    if current_cycle['shares'] > 0:
        curr_price = df['Close'].iloc[-1]
        curr_ret = (curr_price - avg_price) / avg_price
        current_status = {
            'shares': current_cycle['shares'],
            'avg_price': avg_price,
            'invested': current_cycle['invested'],
            'days': current_cycle['days'],
            'status': current_cycle['status'],
            'current_return': curr_ret * 100
        }
        
    return eq_df, cyc_df, current_status

with st.spinner("알고리즘이 백테스트를 수행 중입니다..."):
    eq_df, cyc_df, current_status = run_backtest(BACKTEST_START, BACKTEST_END, INITIAL_CAPITAL)

# --- 결과 요약 대시보드 ---
st.subheader("📊 백테스트 종합 결과")

final_equity = eq_df['Total Equity'].iloc[-1]
total_return = (final_equity / INITIAL_CAPITAL) - 1
total_days = (eq_df.index[-1] - eq_df.index[0]).days
cagr = (final_equity / INITIAL_CAPITAL) ** (365.25 / total_days) - 1 if total_days > 0 else 0
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
if current_status:
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("진행 기간 (일)", f"{current_status['days']} 일")
    s_col2.metric("현재 투입 금액", f"${current_status['invested']:,.2f}")
    s_col3.metric("보유 수량 및 평단가", f"{current_status['shares']}주 / ${current_status['avg_price']:.2f}")
    
    ret_color = "normal" if current_status['current_return'] >= 0 else "inverse"
    s_col4.metric("현재 수익률", f"{current_status['current_return']:+.2f}%", delta_color=ret_color)
    
    if current_status['status'] == 'buying':
        st.info("🟢 **상태:** 매수 진행 중 (40일 이내 또는 목표 수익 도달 대기 중)")
    else:
        st.error("🔴 **상태:** 타임아웃! 강제 존버 모드 (추가 매수 없이 본전 회복 대기 중)")
else:
    st.success("✅ 현재 모든 사이클이 익절/청산되어 **현금 100%** 보유 중입니다. 다음 RSI 매수 타점을 대기합니다.")

st.divider()

# --- 사이클별 상세 결과 (표) ---
st.subheader("📋 사이클별 매매 상세 내역")
if not cyc_df.empty:
    # 예쁜 형식으로 출력
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
    st.info("아직 매도가 완료된 사이클이 없습니다.")

st.divider()

# --- 자산 성장 곡선 ---
st.subheader("📈 전체 자산 성장 곡선 및 투입 원금 변화")
fig = go.Figure()
# 전체 자산 평가액
fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df['Total Equity'], mode='lines', name='총 자산 (현금+주식)', line=dict(color='#8e44ad', width=2.5)))
# 주식에 투입된 금액 (현금 제외)
fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df['Invested'], mode='lines', name='주식 투입 금액', fill='tozeroy', line=dict(color='#e74c3c', width=1.5, dash='dot')))

fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="달러 ($)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 나만의 매매 일지 ---
st.subheader("📓 나만의 매매 일지 (Trading Journal)")
st.markdown("매매를 진행하며 느낀 점이나 원칙 준수 여부를 자유롭게 기록하세요.")
journal_text = st.text_area("일지 입력", height=200, placeholder="예) 2026-03-01: -10% 밑으로 떨어져서 원칙대로 강제 존버 모드에 돌입했다. 심리적으로 흔들리지만 알고리즘을 믿어보자.")
