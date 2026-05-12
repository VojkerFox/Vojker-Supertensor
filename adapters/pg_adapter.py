# -*- coding: utf-8 -*-
import psycopg2
import pickle
import hashlib
import numpy as np
from psycopg2 import sql

class PostgresAdapter:
    def __init__(self, dbname="vojker_db", user="postgres", password="password", host="localhost", port="5432"):
        """
        Vojker Cpk 3.0 PostgreSQL Adapter - Data Lake & Machine Learning Silo
        """
        self.conn_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        self._initialize_db()

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def _initialize_db(self):
        """Alustaa audit-taulun, koneoppimisen Tensor Vaultin ja SHA256 Audit Ledgerin."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Master-Tester Audit Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_audit_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50),
                    total_errors INTEGER,
                    execution_time_sec FLOAT
                )
            """)
            
            # 2. Machine Learning Tensor Vault
            # Tallentaa (3, 30, 4) tensorit ja lopputuloksen (PnL) JAX-koneoppimista varten
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS supertensor_vault (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol_index INTEGER,
                    direction INTEGER,
                    tensor_bytea BYTEA,
                    outcome_pips FLOAT DEFAULT NULL
                )
            """)

            # 3. UUSI: Trade Audit Ledger (Cpk 3.0 SHA256 kryptografia)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_audit_ledger (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol_index INTEGER,
                    state INTEGER,
                    price FLOAT,
                    rnai FLOAT,
                    action TEXT,
                    sha256_signature TEXT
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"\033[91m[DB ERROR] Could not initialize PostgreSQL: {e}\033[0m")

    def log_test_run(self, status, total_errors, execution_time_sec):
        """Kirjoittaa Master-Testerin tulokset tietokantaan."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO test_audit_log (status, total_errors, execution_time_sec)
                VALUES (%s, %s, %s)
            """, (status, total_errors, execution_time_sec))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"\033[91m[DB ERROR] Failed to save audit log: {e}\033[0m")

    def save_tensor_snapshot(self, symbol_index, direction, single_symbol_tensor):
        """
        Tallentaa yksittäisen (3, 30, 4) tensorin tietokantaan ACTION-hetkellä.
        Tämä kerryttää datamassaa Alpha Forge -optimointia varten.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Muutetaan NumPy/JAX-tensori binäärimuotoon tallennusta varten
            tensor_bytes = pickle.dumps(np.array(single_symbol_tensor))
            
            cursor.execute("""
                INSERT INTO supertensor_vault (symbol_index, direction, tensor_bytea)
                VALUES (%s, %s, %s) RETURNING id
            """, (symbol_index, direction, tensor_bytes))
            
            trade_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            return trade_id
        except Exception as e:
            print(f"\033[91m[DB ERROR] Failed to save tensor snapshot: {e}\033[0m")
            return None

    def update_trade_outcome(self, trade_id, outcome_pips):
        """Päivittää kaupan lopputuloksen tensoriin, jotta tekoäly tietää oliko päätös oikea."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE supertensor_vault
                SET outcome_pips = %s
                WHERE id = %s
            """, (outcome_pips, trade_id))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"\033[91m[DB ERROR] Failed to update outcome: {e}\033[0m")

    def log_trade_decision(self, symbol_index, state, price, rnai, action):
        """
        UUSI: Tallentaa FSM-päätöksen PostgreSQL-tietokantaan SHA256-allekirjoituksella.
        Tämä tekee järjestelmästä kryptografisesti auditoitavan (Bytes don't lie).
        """
        try:
            # Luodaan deterministinen merkkijono allekirjoitusta varten
            raw_data = f"{symbol_index}_{state}_{price:.5f}_{rnai:.4f}_{action}"
            signature = hashlib.sha256(raw_data.encode('utf-8')).hexdigest()

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO trade_audit_ledger (symbol_index, state, price, rnai, action, sha256_signature)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (symbol_index, state, price, rnai, action, signature))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"\033[91m[DB ERROR] Audit Logging failed: {e}\033[0m")