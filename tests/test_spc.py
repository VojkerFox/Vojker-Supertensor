# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adapters.pg_adapter import PostgresAdapter
from scripts.spc_monitor import SPCMonitor

def test_spc_integrity():
    print("=== VOJKER PHASE 5: SPC & DB INTEGRITY SUITE (Cpk 3.0) ===")
    errors = 0

    # VAIHE 1: Tietokantayhteyden testaus
    print("\n[Step 1] Verifying PostgreSQL Connection & Schema...")
    try:
        pg = PostgresAdapter()
        conn = pg._get_connection()
        print("  PASSED: PostgreSQL connection established.")
        conn.close()
    except Exception as e:
        print(f"  FAILED: PostgreSQL connection error: {e}")
        print("  -> Varmista, että PostgreSQL on asennettu ja palvelin on käynnissä.")
        errors += 1

    # VAIHE 2: SPC Matematiikan Determinismi
    print("\n[Step 2] Verifying SPC Mathematical Precision...")
    spc = SPCMonitor(target_usl=3.0)
    
    # Skenaario A: Täydellinen järjestelmä (Keskiarvo 1.5s, minimaalinen hajonta)
    perfect_data = np.array([1.50, 1.51, 1.49, 1.50, 1.50, 1.52, 1.48, 1.50])
    mu_p, sigma_p, ucl_p, lcl_p, cpk_p = spc.calculate_cpk(perfect_data)
    
    if cpk_p >= 3.0:
        print(f"  PASSED: Perfect data yielded Six Sigma Cpk ({cpk_p:.2f} >= 3.0)")
    else:
        print(f"  FAILED: Perfect data Cpk too low: {cpk_p:.2f}")
        errors += 1

    # Skenaario B: Epästabiili järjestelmä (Suuri hajonta, lähellä USL-rajaa)
    unstable_data = np.array([2.5, 2.9, 1.5, 3.2, 2.1, 2.8, 1.9, 3.1])
    mu_u, sigma_u, ucl_u, lcl_u, cpk_u = spc.calculate_cpk(unstable_data)
    
    if cpk_u < 1.0:
        print(f"  PASSED: Unstable data correctly flagged (Cpk {cpk_u:.2f} < 1.0)")
    else:
        print(f"  FAILED: Unstable data got unexpectedly high Cpk: {cpk_u:.2f}")
        errors += 1

    # VAIHE 3: DB Read/Write (Audit Trail)
    print("\n[Step 3] Verifying Audit-Ledger I/O...")
    try:
        # 1. Kirjoitetaan testirivi tietokantaan
        test_status = "VERIFIED (SPC-TEST)"
        pg.log_test_run(test_status, 0, 1.123)
        
        # 2. Haetaan data monitorilla
        data = spc.fetch_latest_metrics(limit=5)
        if data is not None and len(data) > 0:
            print("  PASSED: Successfully wrote and fetched metrics from PostgreSQL.")
        else:
            print("  FAILED: Could not fetch metrics after writing.")
            errors += 1
            
        # 3. Kirurginen siivous (Poistetaan vain äsken luotu testirivi)
        conn = pg._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM test_audit_log WHERE status = %s", (test_status,))
        conn.commit()
        cursor.close()
        conn.close()
        print("  PASSED: Database cleanup successful. No traces left.")
        
    except Exception as e:
        print(f"  FAILED: Audit-Ledger I/O error: {e}")
        errors += 1

    # LOPPUTULOS
    print("\n" + "="*55)
    if errors == 0:
        print("  STATUS: SPC & DB INTEGRITY VERIFIED (Cpk 3.0)")
    else:
        print(f"  STATUS: INTEGRITY BREACH ({errors} errors)")
        sys.exit(errors)
    print("="*55)

if __name__ == "__main__":
    test_spc_integrity()