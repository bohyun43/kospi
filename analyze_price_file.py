# -*- coding: utf-8 -*-
"""실시간 가격 데이터를 분석하여 매매 신호를 생성하는 함수 모듈.

이 모듈에는 ``analyze_price_file`` 함수가 정의되어 있습니다. 기존 거대한
스크립트에서 핵심 로직만 분리한 것으로, 원하는 곳에서 불러서 사용할 수
있도록 구성했습니다.

함수는 분봉 가격 파일을 읽어 이동평균선과 RSI 등을 계산하고, 조건에 따라
매수/매도 시그널을 출력합니다. 정상 동작하려면 아래에 명시된 전역 변수가
미리 정의되어 있어야 합니다. (파일 경로, 전략 파라미터, 상태 변수 등)
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# 필요한 전역 변수
# ---------------------------------------------------------------------------
# 아래 변수들은 호출부에서 미리 선언되어야 하며, 가격 데이터 파일 경로와
# 매매 전략에 필요한 각종 파라미터를 담고 있습니다. 예를 들어
# ``FILE_PATH_TARGET`` 은 누적 분봉 데이터를 저장한 CSV 파일 위치를 가리킵니다.
# ---------------------------------------------------------------------------

# The following globals are expected to be defined by the caller:
#   FILE_PATH_TARGET, DATA_PATH, LAST_ANALYZED_PATH
#   TRANSACTION_FEE, RSI_OVERSOLD, RSI_OVERBOUGHT
#   MIN_HOLDING_SECONDS, STOP_LOSS_RATIO, TRAILING_STOP_RATIO
#   TAKE_PROFIT_RATIO
#   price_data, buy_flag, last_buy, trades
#   last_analyzed_index, signal_cooldown


def analyze_price_file():
    """가격 데이터를 분석해 매매 시그널을 만드는 핵심 함수.

    ``FILE_PATH_TARGET`` 에 누적된 분봉 데이터를 읽어 이동평균선과 RSI를
    계산하고, 조건이 맞으면 전역 변수 ``trades`` 에 매매 내역을 추가합니다.
    마지막으로 분석한 위치는 ``LAST_ANALYZED_PATH`` 파일에 저장하며, 분석
    결과는 ``DATA_PATH`` 에 덮어쓰기 합니다. 호출하는 쪽에서는 이 함수가
    전역 변수를 갱신한다는 점을 유의하세요.
    """
    global price_data, buy_flag, last_buy, trades
    global last_analyzed_index, signal_cooldown

    # 데이터 파일이 없으면 더 이상 진행하지 않습니다.
    if not os.path.exists(FILE_PATH_TARGET):
        return

    df = pd.read_csv(FILE_PATH_TARGET)

    # 필수 컬럼(datetime, cur_prc)이 없으면 분석 자체가 불가능합니다.
    if 'datetime' not in df.columns or 'cur_prc' not in df.columns:
        return

    # 날짜 및 가격 데이터 형식을 정리하고 정렬합니다.
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['cur_prc'] = pd.to_numeric(df['cur_prc'], errors='coerce')
    df = df[['datetime', 'cur_prc']].dropna().sort_values('datetime').reset_index(drop=True)

    # 이전에 분석한 인덱스보다 데이터가 길지 않다면 새로 분석할 부분이 없습니다.
    if len(df) <= last_analyzed_index or len(df) < 40:
        return

    # === 기술적 지표 계산 ===
    # 이동평균선과 볼린저밴드를 준비합니다.
    df['ma5'] = df['cur_prc'].rolling(5).mean()
    df['ma10'] = df['cur_prc'].rolling(10).mean()
    df['ma20'] = df['cur_prc'].rolling(20).mean()
    df['bb_std'] = df['cur_prc'].rolling(20).std()
    df['bb_upper'] = df['ma20'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['ma20'] - (df['bb_std'] * 2)

    # RSI 지표와 단기 변동성 계산
    delta = df['cur_prc'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(21).mean()
    avg_loss = loss.rolling(21).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['price_volatility'] = df['cur_prc'].rolling(10).std() / df['cur_prc'].rolling(10).mean()

    start_idx = max(last_analyzed_index, 1)
    new_data = df.iloc[start_idx:]

    # 더 이상 새 데이터가 없으면 바로 종료합니다.
    if len(new_data) == 0:
        return

    print(f"\U0001f501 새로운 {len(new_data)}개 행 분석 중... (전체: {len(df)}개)")

    # 최근 매매 후 일정 시간은 쿨다운 기간으로 두어 신호 남발을 방지합니다.
    if signal_cooldown > 0:
        signal_cooldown -= 1

    # 이번에 새로 분석하는 구간만 순회합니다.
    for i in range(len(new_data)):
        actual_idx = start_idx + i
        if actual_idx < 21:
            continue  # 지표 계산을 위해 최소 21개 데이터 필요
        prev = df.iloc[actual_idx - 1]
        curr = df.iloc[actual_idx]
        if pd.isna(curr['ma20']) or pd.isna(curr['rsi']) or pd.isna(curr['bb_lower']):
            continue

        if not buy_flag and signal_cooldown == 0:
            # --- 매수 조건 판단 ---
            buy_condition_met = None
            # MA5 > MA10 > MA20 의 정배열 상태에서 전일 MA10 아래였다가 다시 돌파하면 매수
            is_uptrend = curr['ma5'] > curr['ma10'] and curr['ma10'] > curr['ma20']
            is_dip_buy = (prev['cur_prc'] <= prev['ma10'] and curr['cur_prc'] > curr['ma10'])
            if is_uptrend and is_dip_buy and curr['rsi'] < 65:
                buy_condition_met = "정배열 눌림목"
            # RSI가 과매도 구간에서 탈출할 때 매수
            elif prev['rsi'] < RSI_OVERSOLD and curr['rsi'] >= RSI_OVERSOLD and curr['cur_prc'] > prev['cur_prc']:
                buy_condition_met = "RSI 과매도 탈출"
            # 가격이 볼린저 밴드 하단을 이탈했다가 다시 상향 돌파하면 매수
            elif prev['cur_prc'] < prev['bb_lower'] and curr['cur_prc'] > curr['bb_lower']:
                buy_condition_met = "볼린저밴드 반등"

            if buy_condition_met:
                # 매수 포지션 진입
                buy_flag = True
                last_buy = {
                    'time': curr['datetime'],
                    'price': curr['cur_prc'],
                    'high_price': curr['cur_prc'],  # 이후 최고가 추적용
                }
                signal_cooldown = 15  # 일정 기간 동안 추가 매수 신호 차단
                print(
                    f"[{curr['datetime']}] \U0001f7e2 매수 시그널 ({buy_condition_met}) @ {curr['cur_prc']:.2f} "
                    f"(RSI: {curr['rsi']:.1f}, MA5: {curr['ma5']:.1f})"
                )

        elif buy_flag and last_buy:
            # --- 보유 중일 때 매도 조건 체크 ---
            hold_time = (curr['datetime'] - last_buy['time']).total_seconds()
            last_buy['high_price'] = max(last_buy['high_price'], curr['cur_prc'])

            if hold_time >= MIN_HOLDING_SECONDS:
                sell_condition_met = None
                # 손절선 하회 시 즉시 매도
                if curr['cur_prc'] < last_buy['price'] * STOP_LOSS_RATIO:
                    sell_condition_met = f"고정손절({STOP_LOSS_RATIO*100-100:.1f}%)"
                # 고점 대비 일정 비율 이하로 떨어지면 이익 보존 차원에서 매도
                elif curr['cur_prc'] < last_buy['high_price'] * TRAILING_STOP_RATIO:
                    sell_condition_met = f"이익보존({TRAILING_STOP_RATIO*100-100:.1f}%)"
                # 목표 수익 도달 시 매도
                elif curr['cur_prc'] > last_buy['price'] * TAKE_PROFIT_RATIO:
                    sell_condition_met = f"목표익절({TAKE_PROFIT_RATIO*100-100:.1f}%)"

                # 데드크로스 발생 또는 RSI 과매수권 하향 돌입 시
                is_dead_cross = prev['ma5'] >= prev['ma20'] and curr['ma5'] < curr['ma20']
                is_rsi_turn = prev['rsi'] > RSI_OVERBOUGHT and curr['rsi'] <= RSI_OVERBOUGHT
                if (is_dead_cross or is_rsi_turn) and not sell_condition_met:
                    sell_condition_met = "기술적매도"

                if sell_condition_met:
                    buy_flag = False
                    profit = curr['cur_prc'] * (1 - TRANSACTION_FEE) - last_buy['price'] * (1 + TRANSACTION_FEE)
                    ret = round(profit / (last_buy['price'] * (1 + TRANSACTION_FEE)) * 100, 2)
                    trades.append(
                        {
                            'buy_time': last_buy['time'],
                            'buy_price': last_buy['price'],
                            'sell_time': curr['datetime'],
                            'sell_price': curr['cur_prc'],
                            'profit': profit,
                            'return(%)': ret,
                            'hold_seconds': hold_time,
                        }
                    )

                    print(
                        f"[{curr['datetime']}] \U0001f534 매도 시그널 ({sell_condition_met}) @ {curr['cur_prc']:.2f} "
                        f"수익: {profit:.0f}원 ({ret:.2f}%) 보유: {hold_time/60:.1f}분"
                    )
                    last_buy = {}

    # 분석 완료 후 마지막 인덱스를 저장하여 다음 실행 시 이어서 분석할 수 있게 합니다.
    last_analyzed_index = len(df) - 1

    try:
        with open(LAST_ANALYZED_PATH, 'w') as f:
            f.write(str(last_analyzed_index))
    except Exception as e:
        print(f"\u26a0\ufe0f 분석 위치 저장 실패: {e}")

    # 20개 간격으로 현재 지표 상태를 출력하여 모니터링에 활용합니다.
    if len(df) % 20 == 0:
        latest = df.iloc[-1]
        volatility = latest.get('price_volatility', 0) * 100
        print(
            f"\U0001f4ca 현재 상태 - 가격: {latest['cur_prc']:.0f}, MA5: {latest['ma5']:.0f}, "
            f"MA20: {latest['ma20']:.0f}, RSI: {latest['rsi']:.1f}, 변동성: {volatility:.2f}%"
        )

    # 갱신된 데이터프레임을 전역 변수와 파일로 저장합니다.
    price_data = df
    df.to_csv(DATA_PATH, index=False)