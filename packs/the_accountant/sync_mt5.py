# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import psycopg2
from psycopg2.extras import execute_values
import sys

# --- TÄRKEÄÄ: PÄIVITÄ NÄMÄ ---
DB_PARAMS = {
    "dbname": "vojker_db", # Tarkista pgAdminista onko tämä oikein
    "user": "postgres",
    "password": "password", # <--- TÄMÄ ON PAKOLLINEN
    "host": "localhost"
}

def run_instrument_sync():
    # Pakotetaan tulostus näkyviin heti (Visual Management)
    print("--- 1. KÄYNNISTETÄÄN SYNKRONOINTI ---", flush=True)
    
    if not mt5.initialize():
        print(f"VIRHE: MT5 alustus epäonnistui. Virhekoodi: {mt5.last_error()}", flush=True)
        return

    print("--- 2. MT5 YHTEYS OK. HAETAAN SYMBOLIT... ---", flush=True)
    symbols = mt5.symbols_get()
    
    if symbols is None or len(symbols) == 0:
        print("VIRHE: Symboleita ei löytynyt. Onko MT5 Market Watch tyhjä?", flush=True)
        mt5.shutdown()
        return
    
    print(f"Löytyi {len(symbols)} instrumenttia. Luodaan batch-paketti...", flush=True)

    data_to_sync = []
    skipped_count = 0

    for s in symbols:
        # POKA-YOKE: Käytetään MT5 Python API:n virallisia attribuuttinimiä
        # Käytetään try-except -rakennetta, jos jollain symbolilla on puutteellisia tietoja
        try:
            data_to_sync.append((
                s.name,
                float(s.trade_contract_size),
                float(s.trade_tick_size),    # KORJATTU: trade_tick_size
                float(s.trade_tick_value),   # KORJATTU: trade_tick_value
                float(s.point),
                int(s.digits)
            ))
        except (AttributeError, TypeError):
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(f"Huomio: Ohitettiin {skipped_count} puutteellista symbolia.", flush=True)

    print(f"--- 3. YHDISTETÄÄN TIETOKANTAAN: {DB_PARAMS['dbname']} ---", flush=True)
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Batch Upsert: Päivittää olemassa olevat, lisää uudet.
        upsert_query = """
            INSERT INTO instrument_registry 
            (symbol, contract_size, tick_size, tick_value, point_value, digits)
            VALUES %s
            ON CONFLICT (symbol) DO UPDATE SET
                contract_size = EXCLUDED.contract_size,
                tick_size = EXCLUDED.tick_size,
                tick_value = EXCLUDED.tick_value,
                point_value = EXCLUDED.point_value,
                digits = EXCLUDED.digits,
                last_sync = CURRENT_TIMESTAMP;
        """
        
        execute_values(cur, upsert_query, data_to_sync)
        conn.commit()
        print(f"--- 4. VALMIS! {len(data_to_sync)} RIVIÄ PÄIVITETTY KANTAAN. ---", flush=True)
        
    except Exception as e:
        print(f"TIETOKANTAVIRHE: {e}", flush=True)
    finally:
        if conn:
            cur.close()
            conn.close()
        mt5.shutdown()

if __name__ == "__main__":
    run_instrument_sync()