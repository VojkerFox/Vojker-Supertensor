# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import time

def open_test_trade(symbol, order_type="BUY", lot=0.01):
    """Avaa kaupan ja palauttaa position tiketin."""
    if not mt5.initialize(): return None
    
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    
    price = tick.ask if order_type == "BUY" else tick.bid
    type_mt5 = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": type_mt5,
        "price": price,
        "magic": 123456,
        "comment": "VOJKER OPEN TEST",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"FAILED OPEN: {result.comment} (Koodi: {result.retcode})")
        return None
    
    print(f"SUCCESS: {symbol} Avattu. Tiketti: {result.order}")
    return result.order

def close_test_trade(ticket):
    """Sulkee position tiketin perusteella."""
    # Haetaan avoin positio tiketin numerolla
    position = mt5.positions_get(ticket=ticket)
    if not position:
        print(f"Virhe: Positiota {ticket} ei löydy.")
        return None
    
    pos = position[0]
    symbol = pos.symbol
    lot = pos.volume
    type_mt5 = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": type_mt5,
        "position": ticket,
        "price": price,
        "magic": 123456,
        "comment": "VOJKER CLOSE TEST",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"FAILED CLOSE: {result.comment}")
    else:
        print(f"SUCCESS: {symbol} Suljettu. Voitto/Tappio: {pos.profit} USD")
    return result