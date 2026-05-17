# 🗺️ VOJKER-ARKITEHTUURI: 5-AKSELINEN KANTAGEOMETRIA & MUISTISOPIMUS (SOP-001)

**Tila:** LUKITTU (DoD Phase 1)  
**Standardiversio:** CPK 3.0 Verified  
**Haara:** accountorpro

Tämä dokumentti määrittelee Vojker-moottorin suojatensorien kiinteät muodot, tyyppiturvallisuuden ja muistinkierrätyssäännöt. Yksikään ajonaikainen tai toiminnallinen transformaatiomoduuli ei saa rikkoa näitä sääntöjä.

## 1. Matemaattinen 5-akselinen kantageometria (Static Shapes)

XLA-kääntäjän operaatioiden fuusion (_operation fusion_) ja rajun `vmap`-rinnakkaistamisen mahdollistamiseksi markkinadata alistetaan kiinteään 5D-suojatensoriin. Dynaamiset muodot on evätty Tracing-virheiden estämiseksi.

$$\mathbf{X} \in \mathbb{R}^{b \times 8 \times 30 \times 4 \times 4}$$

- **Akseli 0 ($b$):** Batch / Skenaarioulottuvuus (Kiinteä koko simulaatioille, historiallisille pätkille tai Monte Carlo -stressitesteille).
- **Akseli 1 ($n$):** Wolfpack-instrumentit (Aina tasan 8 symbolia suorituskykysymmetrian vuoksi).
- **Akseli 2 ($h$):** Historiadepth (Aina tasan 30 kynttilää aikasarjakontekstina).
- **Akseli 3 ($d$):** Core-ominaisuudet (4 ulottuvuutta: 0=Open, 1=High, 2=Low, 3=Close).
- **Akseli 4 ($c$):** Kontekstikerrokset (4 ulottuvuutta: 0=M1 hinta, 1=M5 rakenne, 2=RNAI Advisor, 3=BOS rajat).

**Järjestelmän tilatensori:**
$$\mathbf{S}_{\text{fsm}} \in \mathbb{Z}^{b \times 8}$$
Sisältää kunkin skenaarion ja symbolin senhetkisen Panama FSM -tilakoodin (`1=IDLE`, `3=ACTION`) 32-bittisenä kokonaislukuna.

## 2. Muistisopimus: Puskurin lahjoitus (Buffer Donation)

Erityissyiden varianssin (_Special Cause Variation_) ja Pythonin automaattisen roskienkeruun (_Garbage Collection Stop-the-World_) aiheuttamien latenssipiikkien eliminoimiseksi tuotannon 16Hz ydinloopissa sovelletaan puskurien kierrätystä.

- **Sääntö:** JAX-pääfunktiossa FSM-tilatensori ($\mathbf{S}_{\text{fsm}}$) on merkittävä lahjoitettavaksi argumentiksi käyttäen kääntäjän `donate_argnums` -parametria.
- **Vaikutus:** Uutta muistia ei varata GPU/CPU-kiihdyttimellä 16Hz loopin aikana, vaan uusi FSM-tila kirjoitetaan suoraan vanhan päälle prosessorin nopeassa välimuistissa (_hot memory_).

## 3. Tyyppiregression Esto & Alustatakuu (MT5 / cTrader)

Finanssidatan millintarkka pyöristysvarmuus ja alustariippumattomuus (MetaTrader 5 / cTrader FIX-syötteet) taataan pakottamalla globaali 64-bittisyys heti alustuksessa:
`jax.config.update("jax_enable_x64", True)`

Tämä estää kahden eri tarkkuusluokan kohtaamisen live-loopissa, mikä eliminoi lennosta tapahtuvat JIT-uudelleenkäännökset (_re-compilation bubble_).
