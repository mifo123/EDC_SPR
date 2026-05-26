# Navrhnuté zadanie seminárnej práce

**Názov:** Hierarchický fuzzy expertný systém na prioritizáciu Z-DNA-formujúcich regiónov v genómových dátach

**Východisko:** Pri praktickej analýze Z-DNA-formujúcich regiónov vzniká rozhodovací problém: výpočtové nástroje môžu vytvoriť veľa kandidátov, ale experimentálna validácia alebo manuálna kurácia je kapacitne obmedzená.

**Cieľ seminárnej práce:** Navrhnúť, implementovať a experimentálne overiť fuzzy expertný systém, ktorý zoradí kandidátne Z-DNA-formujúce regióny podľa priority pre experimentálnu validáciu. Riešenie je určené ako univerzálna rozhodovacia vrstva nad genómovými kandidátmi. Vstup `zdna_score` je normalizovaný do intervalu 0-1 a dĺžka kandidátov je 8-400 bp. Systém má kombinovať sekvenčný signál, regulačný kontext, typ evidencie, riziko datasetového biasu a praktickú validovateľnosť.

**Rozhodovací výstup:** Každý kandidát je zaradený do jednej z front:

- `A_validovat`: kandidát vhodný pre prvú vlnu validácie,
- `B_prioritne_preskumat`: kandidát vhodný na bližšie preskúmanie,
- `C_manualna_kuracia`: kandidát vyžadujúci manuálnu kontrolu,
- `D_odlozit`: kandidát s nízkou prioritou alebo vysokým rizikom biasu.

**Požadované časti riešenia:**

1. formalizácia rozhodovacieho problému a väzba na analýzu lokálnych štruktúr DNA,
2. návrh vstupných premenných, fuzzy množín a pravidlovej bázy,
3. implementácia Mamdaniho inferenčného mechanizmu a porovnávacieho Takagiho-Sugenovho fuzzy modelu v jazyku Python,
4. príprava označeného benchmarku 320 kandidátov rozdeleného na tréningovú, validačnú a testovaciu časť,
5. porovnanie viacerých rozhodovacích stratégií vrátane baseline, lineárneho modelu, Mamdaniho modelov a Takagiho-Sugenovho modelu,
6. vyhodnotenie najlepšieho prístupu pomocou MAE, korelácie, presnosti validačnej fronty a top-k metriky,
7. ablačná analýza prínosu CpX kontextu, TSS vzdialenosti, biasu a validovateľnosti,
8. odhad intervalovej neistoty výsledku opakovanou perturbáciou vstupných premenných,
9. externé overenie na experimentálne validovaných Shin et al. dátach pomocou reálnych per-locus overlay súborov,
10. diskusia validácie, verifikácie, obmedzení a ďalších možných vylepšení systému,

## Väzba zadania na osnovu predmetu

| Téma osnovy predmetu | Pokrytie v seminárnej práci | Konkrétny výstup |
|---|---|---|
| 1. Počítačová podpora rozhodovania, databázové systémy, vytěžení dat z databází | Prioritizácia kandidátnych genómových regiónov z tabuľkových a overlay dát. | `data/sample_candidates.csv`, `data/shin_real_candidates.csv`, finálny ranking kandidátov. |
| 2. Formalizácia mentálnych expertných modelov | Biologické a validačné úvahy sú prevedené na vstupy, fronty a pravidlá. | Definícia front `A` až `D`, premenné `z_signal`, `regulatory_context`, `bias_risk`, `validation_feasibility`. |
| 3. Jazykové modelovanie a fuzzy reprezentácia znalostí | Vstupy sú reprezentované jazykovými hodnotami nízka, stredná a vysoká. | Funkcie príslušnosti a pravidlová báza Mamdaniho aj Takagiho-Sugenovho modelu. |
| 4. Expertné diagnostické systémy | Systém diagnostikuje prioritu a riziko kandidáta pre validáciu. | Vysvetliteľné zaradenie do validačnej alebo kurátorskej fronty. |
| 5. Teória fuzzy množín a pokročilé aplikácie fuzzy logiky | Použité sú trapézové fuzzy množiny a viacvrstvové fuzzy rozhodovanie. | `outputs/membership_functions.pdf`, opis fuzzy množín v paperi. |
| 6. Inferenčné mechanizmy Mamdani a Takagi-Sugeno | Porovnáva sa hierarchický Mamdaniho model s Takagiho-Sugenovým variantom. | Experimenty E4 až E7 a tabuľka porovnania modelov. |
| 7. Pokročilé metódy znalostného inžinierstva | Pravidlá sú iterované, auditované a hodnotené abláciou vstupov. | `outputs/rule_coverage_summary.csv`, `outputs/rules_audit.txt`, ablačná analýza. |
| 8. Systém LFLC a jeho využitie | Pravidlová báza je formulovaná tak, aby bola konceptuálne prenositeľná do LFLC štýlu. | Jazykové premenné, fuzzy pravidlá a diskusia možnosti prepisu do LFLC. |
| 9. Fuzzy transformácia a analýza dátových radov | Téma je pokrytá ako navrhnuté rozšírenie pre profil priority pozdĺž chromozómu. | Návrh budúceho rozšírenia v diskusii. |
| 10. Analýza dát pre zostavenie parametrov expertného systému | Parametre sú odvodené z benchmarku, splitov, CpX/TSS anotácií a Shin validácie. | Tréningový, validačný a testovací split, externý Shin dataset. |
| 11. Syntéza znalostných systémov | Navrhnutá je viacvrstvová štruktúra znalostného systému. | Vrstvy biologická plausibilita, dôvera v evidenciu a finálna priorita. |
| 12. Zostavenie znalostnej bázy a jej testovanie | Znalostná báza je testovaná na benchmarku, abláciách a rule coverage audite. | Tabuľky experimentov, rule coverage a intervalová neistota. |
| 13. Aplikácie expertných systémov pre podporu rozhodovania | Výsledok podporuje výber kandidátov pre experimentálnu validáciu. | `outputs/priority_ranking.csv` a validačné fronty. |
| 14. Validácia a verifikácia expertného systému | Model je overený na testovacom splite a externých Shin dátach s intervalmi spoľahlivosti. | Shin validácia, konfúzna matica, bootstrap 95 % intervaly. |
