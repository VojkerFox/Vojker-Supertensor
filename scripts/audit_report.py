# -*- coding: utf-8 -*-
import psycopg2
from psycopg2.extras import RealDictCursor
import sys
import os

# Lisätään polku
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_report(limit=10):
    print("\n" + "="*80)
    print("   VOJKER AUDIT LAYER: DECISION LOG & SHA256 CHAIN")
    print("="*80)
    
    try:
        # Yhteys kantaan
        conn = psycopg2.connect(
            dbname="vojker_db",
            user="postgres",
            password="password", # Varmista että tämä on oikein
            host="localhost"
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Deterministinen kysely oikeasta taulusta
        query = """
            SELECT timestamp, symbol_index, state, price, rnai, action, sha256_signature
            FROM trade_audit_ledger
            ORDER BY timestamp DESC
            LIMIT %s;
        """
        cur.execute(query, (limit,))
        rows = cur.fetchall()

        if not rows:
            print("   Ei tallennettuja päätöksiä kannassa.")
        else:
            print(f"{'AIKA':<20} | {'IDX':<3} | {'HINTA':<10} | {'RNAI':<8} | {'SHA256 SIGNATURE'}")
            print("-" * 80)
            for r in rows:
                # Korjattu: sarakkeen nimi on 'timestamp'
                t_str = r['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                # Korjattu: sarakkeen nimi on 'sha256_signature'
                sig_short = r['sha256_signature'][:24] + "..." if r['sha256_signature'] else "N/A"
                
                print(f"{t_str:<20} | {r['symbol_index']:<3} | {r['price']:<10.5f} | {r['rnai']:<8.4f} | {sig_short}")

    except Exception as e:
        print(f"\033[91m[ERROR] Raportointi epäonnistui: {e}\033[0m")
    finally:
        if 'conn' in locals(): conn.close()
    print("="*80 + "\n")

if __name__ == "__main__":
    run_report()