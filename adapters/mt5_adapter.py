# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import jax.numpy as jnp
import numpy as np

WOLFPACK_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]

class MT5Adapter:
    def __init__(self, symbols=WOLFPACK_SYMBOLS, bars=30):
        self.symbols = symbols
        self.bars = bars

    def get_wolfpack_tensors(self, quantum_context=None):
        raw_symbol_containers = []
        symbol_aggressions = []
        
        # VAIHE 1: Kerätään raakadata ja lasketaan symbolikohtainen aggressio
        for s in self.symbols:
            r_m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, self.bars)
            r_m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, self.bars)
            
            if r_m1 is None or r_m5 is None or len(r_m1) < self.bars:
                return None
            
            # OHLC Data M1 (K=0) - Pidetään raakana nykyhetkenä
            m1_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m1], dtype=np.float32)

            # DETERMINISTINEN ANKKURI (K=1): Lasketaan keskiarvo raakadatan sijaan (Divergenssi)
            m5_avg = np.mean([r[4] for r in r_m5]) 
            m5_anchor_data = np.ones_like(m1_data) * m5_avg 
            
            # RNAI PROXY: (Close - Open) * TickVolume
            last_candle = r_m1[-1]
            net_aggression = (last_candle[4] - last_candle[1]) * last_candle[5] 
            symbol_aggressions.append(net_aggression)
            
            # Lisätään ankkuroitu data listaan
            raw_symbol_containers.append((m1_data, m5_anchor_data))

        # VAIHE 2: Markkinan keskiarvon laskenta
        market_avg_aggression = np.mean(symbol_aggressions)
        
        # VAIHE 3: Rakennetaan lopullinen 4D-supertensori (8, 3, 30, 4)
        all_symbol_data = []
        for i, (m1_data, m5_anchor) in enumerate(raw_symbol_containers):
            # Lasketaan suhteellinen aggressio (RNAI)
            rnai_val = symbol_aggressions[i] - market_avg_aggression
            
            # K=2 täytetään RNAI-arvolla
            q_data = np.ones_like(m1_data, dtype=np.float32) * rnai_val
            
            # Pinotaan: K=0 (M1), K=1 (M5 Anchor), K=2 (RNAI Context)
            symbol_stack = np.stack([m1_data, m5_anchor, q_data], axis=0)
            all_symbol_data.append(symbol_stack)
            
        return jnp.array(all_symbol_data) if len(all_symbol_data) == 8 else None

    def calculate_lot_size(self, symbol, sl_distance_pips, risk_pct=0.0075):
        """
        Cpk 3.0 Kuminauha-positiointi.
        Laskee Lot Sizen siten, että tappio on aina tasan 0.75 % tilin pääomasta.
        Käyttää korjattuja MT5-attribuutteja: trade_tick_size ja trade_tick_value.
        """
        account = mt5.account_info()
        if account is None:
            return 0.01 
        
        balance = account.balance
        risk_amount = balance * risk_pct
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.01

        # MetaTrader 5 API:n oikeat attribuutit
        point = symbol_info.point
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value

        if tick_size == 0 or tick_value == 0:
            return 0.01

        # Lasketaan piste-etäisyys (pips -> points)
        sl_points = sl_distance_pips * 10 * point
        
        if sl_points == 0:
            return 0.01
            
        # Kuminauhakaava: Joustava lot-koko perustuen Salaman laatikkoon
        lots = risk_amount / ((sl_points / tick_size) * tick_value)
        lots = round(lots, 2)
        
        return max(symbol_info.volume_min, min(symbol_info.volume_max, lots))

    def execute_market_order(self, symbol, direction, lot_size, entry_price, sl_price):
        """
        Lähettää varsinaisen toimeksiannon MT5-rajapintaan DYNAAMISELLA täyttötavalla.
        Korjaa TRADE_RETCODE_TRADE_DISABLED (10017) virheen filling_mode-tunnistuksella.
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None

        # --- DYNAAMINEN FILLING MODE (Cpk 3.0 Fix) ---
        filling_mode = getattr(symbol_info, "filling_mode", 0)

        if filling_mode & getattr(mt5, "SYMBOL_FILLING_MODE_IOC", 2) or filling_mode & getattr(mt5, "SYMBOL_FILLING_IOC", 2):
            filling_type = mt5.ORDER_FILLING_IOC
        elif filling_mode & getattr(mt5, "SYMBOL_FILLING_MODE_FOK", 1) or filling_mode & getattr(mt5, "SYMBOL_FILLING_FOK", 1):
            filling_type = mt5.ORDER_FILLING_FOK
        else:
            filling_type = mt5.ORDER_FILLING_RETURN

        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "sl": float(sl_price),
            "deviation": 10,
            "magic": 0x7633f8, 
            "comment": "Vojker 0.75% Risk",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\033[92m[TRADE DONE] {symbol} {'BUY' if direction == 1 else 'SELL'} {lot_size} lots executed.\033[0m")
            return result
        else:
            retcode = getattr(result, "retcode", "N/A")
            print(f"\033[91m[TRADE ERROR] Failed to execute {symbol}. Retcode: {retcode}\033[0m")
            return result

    def close_position(self, symbol):
        """
        EXIT PROTOCOL: Etsii ja sulkee kaikki Vojker-magicilla merkityt positiot.
        Varmistaa deterministisen poistumisen markkinalta.
        """
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return False
            
        closed_any = False
        for position in positions:
            if position.magic != 0x7633f8:
                continue
                
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
                
            type_close = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price_close = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            symbol_info = mt5.symbol_info(symbol)
            filling_mode = getattr(symbol_info, "filling_mode", 0)
            if filling_mode & getattr(mt5, "SYMBOL_FILLING_MODE_IOC", 2) or filling_mode & getattr(mt5, "SYMBOL_FILLING_IOC", 2):
                filling_type = mt5.ORDER_FILLING_IOC
            elif filling_mode & getattr(mt5, "SYMBOL_FILLING_MODE_FOK", 1) or filling_mode & getattr(mt5, "SYMBOL_FILLING_FOK", 1):
                filling_type = mt5.ORDER_FILLING_FOK
            else:
                filling_type = mt5.ORDER_FILLING_RETURN

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": position.ticket,
                "volume": position.volume,
                "type": type_close,
                "price": price_close,
                "deviation": 10,
                "magic": 0x7633f8,
                "comment": "Vojker Exit Protocol",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                 print(f"\033[94m[EXIT DONE] {symbol} position {position.ticket} closed.\033[0m")
                 closed_any = True
            else:
                 retcode = getattr(result, "retcode", "N/A")
                 print(f"\033[91m[EXIT ERROR] Failed to close {symbol} position {position.ticket}. Retcode: {retcode}\033[0m")
                 
        return closed_any