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

