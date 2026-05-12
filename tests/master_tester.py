# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time

# --- CPK 3.0 PATH FIX: Pakotetaan projektin juuri hakupolun kärkeen ---
# Lasketaan polku: ollaan kansiossa 'tests', joten juuri on yksi taso ylempänä
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Lisätään juuri polkuun ensimmäiseksi (0), jotta adapters-kansio löytyy
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NYT vasta importataan adapteri, kun polku on takuulla korjattu
try:
    from adapters.pg_adapter import PostgresAdapter
except ImportError as e:
    print(f"\033[91m[CRITICAL] Adapters not found at {project_root}. Error: {e}\033[0m")
    sys.exit(1)

def run_master_tester():
    print("="*75)
    print("   VOJKER SUPERTENSOR: MASTER-TESTER (Cpk 3.0 Bi-directional)")
    print("   Executing Full Integrity Suite...")
    print("="*75)

    # Kaikki 5 vaihetta auditointia varten
    tests = [
        ("Phase 1: MT5 Adapter & RNAI & Positioning", "tests/test_adapter.py"),
        ("Phase 2: Triple Quantile Gate (Logic)", "tests/test_logic.py"),
        ("Phase 3: Panama FSM (Bi-directional)", "tests/test_fsm.py"),
        ("Phase 4: Triage Engine & Trader HUD", "tests/test_triage_engine.py"),
        ("Phase 5: DB & SPC Analytics", "tests/test_spc.py"),
        
        ("phase 6: Account Audit (Broker Connection)", "tests/test_account_audit.py")
    ]

    total_errors = 0
    start_time = time.time()

    for name, path in tests:
        print(f"\n\033[93m>>> STARTING: {name} ({path})\033[0m")
        print("-" * 75)
        
        try:
            # Ajetaan testit projektin juuresta (cwd=project_root)
            # Käytetään sys.executablea, jotta .venv pysyy käytössä
            result = subprocess.run([sys.executable, path], cwd=project_root)
            
            if result.returncode != 0:
                print(f"\033[91m[!] CRITICAL FAILURE: {name} (Exit Code: {result.returncode})\033[0m")
                total_errors += 1
            else:
                print(f"\033[92m[OK] {name} PASSED.\033[0m")
        except Exception as e:
            print(f"\033[91m[!] CRITICAL FAILURE: Execution error: {e}\033[0m")
            total_errors += 1

    execution_time = time.time() - start_time
    print("\n" + "="*75)
    if total_errors == 0:
        print(f"   \033[92mMASTER STATUS: ALL SYSTEMS VERIFIED (Cpk 3.0)\033[0m")
    else:
        print(f"   \033[91mMASTER STATUS: INTEGRITY BREACH DETECTED ({total_errors} tests failed)\033[0m")
    print("="*75)
    
    return execution_time

if __name__ == "__main__":
    # Alustetaan tietokanta-adapteri
    pg_db = PostgresAdapter()
    
    # 30 minuutin automaatioluuppi
    CYCLE_INTERVAL_SEC = 1800 
    
    print("\033[96m[SYSTEM] Master-Tester initialized in CONTINUOUS MODE (30 min intervals).\033[0m")
    
    try:
        while True:
            exec_time = run_master_tester()
            # Tallennetaan loki
            pg_db.log_test_run("VERIFIED" if exec_time else "FAILED", 0, exec_time)
            print(f"\n\033[90m[SLEEP] Next integrity check in 30 minutes. Zzz...\033[0m\n")
            time.sleep(CYCLE_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n\033[91m[SYSTEM] Master-Tester terminated.\033[0m")
