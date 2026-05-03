# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time

# Varmistetaan, että importit toimivat juurikansiosta
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from adapters.pg_adapter import PostgresAdapter

def run_master_tester():
    print("="*70)
    print("   VOJKER SUPERTENSOR: MASTER-TESTER (Cpk 3.0 Bi-directional)")
    print("   Executing Full Integrity Suite...")
    print("="*70)

    tests = [
        ("Phase 1: MT5 Adapter & RNAI Ingest", "tests/test_adapter.py"),
        ("Phase 2: Triple Quantile Gate (Logic)", "tests/test_logic.py"),
        ("Phase 3: Panama FSM (Bi-directional)", "tests/test_fsm.py"),
        ("Phase 4: Triage Engine & Trader HUD", "tests/test_triage_engine.py"),
        ("Phase 5: DB & SPC Analytics", "tests/test_spc.py") # TÄMÄ ON UUSI
    ]

    total_errors = 0
    start_time = time.time()

    for name, path in tests:
        print(f"\n\033[93m>>> STARTING: {name} ({path})\033[0m")
        print("-" * 70)
        
        try:
            result = subprocess.run([sys.executable, path])
            
            if result.returncode != 0:
                print(f"\033[91m[!] CRITICAL FAILURE: {name} (Exit Code: {result.returncode})\033[0m")
                total_errors += 1
            else:
                print(f"\033[92m[OK] {name} PASSED.\033[0m")
        except FileNotFoundError:
            print(f"\033[91m[!] CRITICAL FAILURE: File not found -> {path}\033[0m")
            total_errors += 1

    end_time = time.time()
    execution_time = end_time - start_time

    # LOPPURAPORTTI
    print("\n" + "="*70)
    if total_errors == 0:
        final_status = "VERIFIED (Cpk 3.0)"
        print(f"   \033[92mMASTER STATUS: ALL SYSTEMS VERIFIED (Cpk 3.0)\033[0m")
        print(f"   \033[97mTotal Execution Time: {execution_time:.2f} seconds\033[0m")
        print("   The Wolfpack is fully operational and ready for deployment.")
    else:
        final_status = f"BREACH ({total_errors} errors)"
        print(f"   \033[91mMASTER STATUS: INTEGRITY BREACH DETECTED ({total_errors} tests failed)\033[0m")
        print("   System is NOT ready. Check the logs above.")
    print("="*70)
    
    return final_status, total_errors, execution_time

if __name__ == "__main__":
    # 1. Alustetaan PostgreSQL-yhteys (muuta tunnukset pg_adapter.py -tiedostoon tarvittaessa)
    pg_db = PostgresAdapter()
    
    # 2. 30 Minuutin Automaatioluuppi (1800 sekuntia)
    CYCLE_INTERVAL_SEC = 1800 
    
    print("\033[96m[SYSTEM] Master-Tester initialized in CONTINUOUS MODE (30 min intervals).\033[0m")
    print("\033[96m[SYSTEM] Press Ctrl+C to terminate.\033[0m")
    
    try:
        while True:
            # Ajetaan koko testipatteristo
            status, errors, exec_time = run_master_tester()
            
            # Tallennetaan tulokset PostgreSQL-tietokantaan
            pg_db.log_test_run(status, errors, exec_time)
            
            # Odotetaan 30 minuuttia ennen seuraavaa sykliä
            print(f"\n\033[90m[SLEEP] Next integrity check in 30 minutes. Zzz...\033[0m\n")
            time.sleep(CYCLE_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        print("\n\033[91m[SYSTEM] Master-Tester continuous loop terminated by user.\033[0m")