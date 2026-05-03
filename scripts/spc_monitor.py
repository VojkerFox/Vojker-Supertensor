# -*- coding: utf-8 -*-
"""
VOJKER SPC MONITOR (Statistical Process Control)
Purpose: Analyzes PostgreSQL audit logs to verify Cpk 3.0 stability continuously.
"""
import sys
import os
import numpy as np

# Varmistetaan importit juuresta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adapters.pg_adapter import PostgresAdapter

class SPCMonitor:
    def __init__(self, target_usl=5.0):
        """
        Alustetaan SPC Monitor.
        target_usl = Upper Specification Limit (esim. testien on pakko mennä läpi alle 5 sekunnissa).
        """
        self.pg = PostgresAdapter()
        self.target_usl = target_usl
        self.target_lsl = 0.0 # Aika ei voi olla negatiivinen

    def fetch_latest_metrics(self, limit=100):
        """Hakee viimeisimmät onnistuneet ajot tietokannasta."""
        try:
            conn = self.pg._get_connection()
            cursor = conn.cursor()
            # Haetaan vain onnistuneet ajot, jotta voimme analysoida koneen normaalia sykettä
            cursor.execute("""
                SELECT execution_time_sec 
                FROM test_audit_log 
                WHERE status LIKE '%%VERIFIED%%' 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return None
            return np.array([row[0] for row in rows])
        except Exception as e:
            print(f"\033[91m[SPC ERROR] DB Fetch failed: {e}\033[0m")
            return None

    def calculate_cpk(self, data):
        """Laskee Cpk ja Cp -arvot datalle."""
        mu = np.mean(data)
        sigma = np.std(data, ddof=1) # Otoksen keskihajonta
        
        if sigma == 0:
            sigma = 0.0001 # Estetään nollalla jako, jos järjestelmä on epäilyttävän täydellinen
            
        # Cpk on pienempi arvo näistä kahdesta (keskiarvon etäisyys rajoista suhteessa hajontaan)
        cpk_upper = (self.target_usl - mu) / (3 * sigma)
        cpk_lower = (mu - self.target_lsl) / (3 * sigma)
        
        cpk = min(cpk_upper, cpk_lower)
        
        # Kontrollirajat (UCL / LCL)
        ucl = mu + (3 * sigma)
        lcl = max(0, mu - (3 * sigma))
        
        return mu, sigma, ucl, lcl, cpk

    def render_spc_report(self):
        """Generoi teollisen "Trader Command Center" -tyylisen SPC-raportin."""
        data = self.fetch_latest_metrics()
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*75)
        print("   VOJKER STATISTICAL PROCESS CONTROL (SPC) TERMINAL")
        print("   Monitoring Metric: Master-Tester Execution Latency")
        print("="*75)

        if data is None or len(data) < 2:
            print("   \033[93m[WAITING] Not enough data points in PostgreSQL to calculate SPC.\033[0m")
            print("   Please let the Master-Tester run a few cycles first.")
            print("="*75)
            return

        mu, sigma, ucl, lcl, cpk = self.calculate_cpk(data)
        
        # Visuaalinen status
        if cpk >= 3.0:
            status_color = "\033[92m" # Vihreä (Six Sigma / Cpk 3.0)
            status_text = "SURGICAL PRECISION (Six Sigma Verified)"
        elif cpk >= 1.33:
            status_color = "\033[93m" # Keltainen (Hyväksyttävä, mutta ei Cpk 3.0)
            status_text = "ACCEPTABLE (Requires Optimization)"
        else:
            status_color = "\033[91m" # Punainen (Epästabiili)
            status_text = "UNSTABLE (Process out of control limits)"

        print(f"   Sample Size (N): {len(data)} latest cycles")
        print(f"   Target Limit (USL): {self.target_usl:.4f} sec")
        print("-" * 75)
        print(f"   Process Mean (μ):      {mu:.4f} sec")
        print(f"   Process Variance (σ):  {sigma:.6f} sec")
        print(f"   Upper Control (UCL):   {ucl:.4f} sec")
        print(f"   Lower Control (LCL):   {lcl:.4f} sec")
        print("-" * 75)
        print(f"   \033[97mCURRENT Cpk SCORE:     {cpk:.4f}\033[0m")
        print(f"   SYSTEM STATUS:         {status_color}{status_text}\033[0m")
        print("="*75)
        
        # Analyysi
        if cpk < 3.0:
            print("\033[93m   [DIAGNOSTICS] Järjestelmä ei ole Cpk 3.0 tasolla.\033[0m")
            if mu + (3*sigma) > self.target_usl:
                print("   -> Syy: Variaatio (σ) on liian suuri tai keskiarvo on liian lähellä USL-rajaa.")
        else:
            print("\033[92m   [DIAGNOSTICS] Järjestelmä toimii deterministisellä Cpk 3.0 tasolla.\033[0m")
            print("   -> Prosessi on täysin ennakoitavissa ja hallinnassa.")
        print("="*75)

if __name__ == "__main__":
    # Asetetaan USL (esim. testien pitää valmistua 3.0 sekunnissa)
    monitor = SPCMonitor(target_usl=3.0)
    monitor.render_spc_report()