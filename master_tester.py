# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time

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
        ("Phase 5: DB & SPC Analytics", "tests/test_spc.py")
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

    print("\n" + "="*70)
    if total_errors == 0:
        final_status = "VERIFIED (Cpk 3.0)"
        print(f"   \033[92mMASTER STATUS: ALL SYSTEMS VERIFIED (Cpk 3.0)\033[0m")
        print(f"   \033[97mTotal Execution Time: {execution_time:.2f} seconds\033[0m")
    else:
        final_status = f"BREACH ({total_errors} errors)"
        print(f"   \033[91mMASTER STATUS: INTEGRITY BREACH DETECTED ({total_errors} tests failed)\033[0m")
    print("="*70)
    
    return final_status, total_errors, execution_time

if __name__ == "__main__":
    pg_db = PostgresAdapter()
    CYCLE_INTERVAL_SEC = 1800 
    
    print("\033[96m[SYSTEM] Master-Tester initialized in CONTINUOUS MODE (30 min intervals).\033[0m")
    
    try:
        while True:
            status, errors, exec_time = run_master_tester()
            pg_db.log_test_run(status, errors, exec_time)
            print(f"\n\033[90m[SLEEP] Next integrity check in 30 minutes. Zzz...\033[0m\n")
            time.sleep(CYCLE_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n\033[91m[SYSTEM] Master-Tester terminated by user.\033[0m")
