# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import sys
import os

# Lisätään projektin juuri polkuun (varmuuden vuoksi)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_account_audit():
    print("="*60)
    print("   VOJKER INFRASTRUCTURE AUDIT: ACCOUNT VERIFICATION")
    print("="*60)

    # 1. Alustetaan yhteys työpöytäsovellukseen
    if not mt5.initialize():
        print(f"\033[91m[FAILED] MT5 Initialization failed. Error code: {mt5.last_error()}\033[0m")
        return

    try:
        # 2. Haetaan tilitiedot
        account_info = mt5.account_info()
        
        if account_info is None:
            print("\033[91m[FAILED] Could not retrieve account info. Are you logged in?\033[0m")
        else:
            # 3. Muotoillaan ja tulostetaan kriittiset tiedot
            print(f"   STATUS:          \033[92mCONNECTED\033[0m")
            print(f"   ACCOUNT NUMBER:  {account_info.login}")
            print(f"   BROKER/SERVER:   {account_info.server}")
            print(f"   PROVIDER:        {account_info.company}")
            print(f"   CURRENCY:        {account_info.currency}")
            print(f"   BALANCE:         {account_info.balance:,.2f} {account_info.currency}")
            print(f"   EQUITY:          {account_info.equity:,.2f} {account_info.currency}")
            print(f"   LEVERAGE:        1:{account_info.leverage}")
            print("-" * 60)
            
            # 4. Varmistetaan kaupankäyntioikeudet (The Teeth Check)
            if account_info.trade_allowed:
                print("   TRADING RIGHTS:  \033[92mALLOWED\033[0m")
            else:
                print("   TRADING RIGHTS:  \033[91mDISABLED (Check your settings)\033[0m")
            
            if account_info.trade_expert:
                print("   ALGO TRADING:    \033[92mENABLED\033[0m")
            else:
                print("   ALGO TRADING:    \033[91mDISABLED (Check the Auto Trading button)\033[0m")

    except Exception as e:
        print(f"\033[91m[CRITICAL ERROR] Audit crashed: {e}\033[0m")
    finally:
        # 5. Suljetaan yhteys siististi
        mt5.shutdown()
        print("="*60)

if __name__ == "__main__":
    run_account_audit()