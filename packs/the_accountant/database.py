# -*- coding: utf-8 -*-
import psycopg2
from psycopg2.extras import RealDictCursor

class WolffDatabase:
    def __init__(self, dbname, user, password, host="localhost"):
        self.params = {"dbname": dbname, "user": user, "password": password, "host": host}

    def get_conn(self):
        return psycopg2.connect(**self.params)

    def fetch_all_symbols(self):
        """Hakee kaikki 2552 instrumenttia JAX-analyysia varten."""
        query = "SELECT symbol, contract_size, tick_value FROM instrument_registry;"
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return cur.fetchall()

    def update_elite_status(self, elite_symbols, scores):
        """
        Poka-Yoke: Nollaa vanhat ja merkitsee uuden 0.8% eliitin.
        elite_symbols: lista symboleista (str)
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                # 1. Nollataan vanha status
                cur.execute("UPDATE instrument_registry SET is_elite = FALSE;")
                
                # 2. Merkataan uudet eliitit
                for symbol in elite_symbols:
                    # Haetaan score jos se on tallessa
                    score = scores.get(symbol, 0.0)
                    cur.execute(
                        "UPDATE instrument_registry SET is_elite = TRUE, compatibility_score = %s WHERE symbol = %s;",
                        (float(score), symbol)
                    )
                conn.commit()