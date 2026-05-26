# Experimenty fuzzy prioritizacie ZFR kandidatov

Vygenerovane: 2026-05-28T09:05:25

Vybrana interpretovatelna konfiguracia na zaklade validacneho splitu: **E7_takagi_sugeno**.

## Validacny vyber modelu

| Poradie | Konfiguracia | MAE | Spearman | Queue acc. | Adjacent acc. | Top-k precision | Quality |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | E7_takagi_sugeno | 4.76 | 0.956 | 0.859 | 1.000 | 0.833 | 86.47 |
| 2 | E3_weighted_linear | 5.00 | 0.947 | 0.797 | 1.000 | 0.889 | 84.42 |
| 3 | E4_fuzzy_permissive | 10.07 | 0.841 | 0.625 | 0.984 | 0.833 | 69.99 |
| 4 | E6_fuzzy_calibrated | 10.18 | 0.872 | 0.562 | 0.984 | 0.778 | 68.65 |
| 5 | E5_fuzzy_conservative | 13.23 | 0.806 | 0.531 | 0.984 | 0.722 | 61.32 |
| 6 | E2_sequence_consensus | 12.05 | 0.292 | 0.484 | 0.922 | 0.500 | 36.42 |
| 7 | E1_zdna_only | 18.50 | -0.101 | 0.281 | 0.750 | 0.278 | 6.96 |

## Testovacie porovnanie

| Poradie | Konfiguracia | MAE | Spearman | Queue acc. | Adjacent acc. | Top-k precision | Quality |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | E7_takagi_sugeno | 5.15 | 0.944 | 0.766 | 1.000 | 0.842 | 82.59 |
| 2 | E3_weighted_linear | 5.69 | 0.943 | 0.719 | 1.000 | 0.842 | 80.79 |
| 3 | E6_fuzzy_calibrated | 8.84 | 0.904 | 0.594 | 1.000 | 0.737 | 71.88 |
| 4 | E4_fuzzy_permissive | 10.55 | 0.863 | 0.531 | 1.000 | 0.789 | 67.60 |
| 5 | E5_fuzzy_conservative | 11.21 | 0.870 | 0.625 | 0.984 | 0.632 | 66.95 |
| 6 | E2_sequence_consensus | 13.28 | 0.409 | 0.312 | 0.875 | 0.526 | 34.82 |
| 7 | E1_zdna_only | 18.54 | 0.085 | 0.328 | 0.703 | 0.263 | 10.33 |

## Ablacna analyza na teste

| Scenar | MAE | Spearman | Queue acc. | Quality | Delta MAE | Quality loss |
|---|---:|---:|---:|---:|---:|---:|
| plny_model | 5.15 | 0.944 | 0.766 | 82.59 | +0.00 | +0.00 |
| bez_CpX_kontextu | 6.02 | 0.942 | 0.719 | 81.90 | +0.88 | +0.69 |
| bez_TSS_vzdialenosti | 8.74 | 0.925 | 0.578 | 73.34 | +3.59 | +9.24 |
| bez_biasu | 8.29 | 0.917 | 0.516 | 70.64 | +3.14 | +11.95 |
| bez_validovatelnosti | 6.15 | 0.935 | 0.688 | 78.23 | +1.01 | +4.35 |

## Intervalova neistota

Priemerna sirka 5-95 % intervalu: 2.99 bodu.
Priemerna stabilita fronty: 0.962.

## Shin validacia

| Model | Recall | Specificity | Balanced acc. | BA 95% CI | Youden J |
|---|---:|---:|---:|---:|---:|
| ZDNABERT HG18 th 0.25 | 0.877 | 0.890 | 0.880 | 0.842-0.923 | 0.770 |
| E7 Z-DNA Hunter + fuzzy best Shin threshold | 0.772 | 0.931 | 0.852 | 0.814-0.888 | 0.704 |
| Z-DNA Hunter model 2 size 10 | 0.726 | 0.973 | 0.850 | 0.817-0.880 | 0.700 |
| Z-DNA Hunter model 1 size 10 | 0.685 | 0.986 | 0.840 | 0.806-0.865 | 0.670 |
| Z-DNA Hunter model 2 size 8 | 0.822 | 0.849 | 0.840 | 0.787-0.879 | 0.670 |
| E7 priority >= B_prioritne_preskumat | 0.811 | 0.836 | 0.823 | 0.773-0.869 | 0.646 |
| Z-DNA Hunter model 2 size 8 score 20% | 0.767 | 0.863 | 0.820 | 0.766-0.861 | 0.630 |
| Z-DNA Hunter model 2 size 8 permissive | 0.877 | 0.740 | 0.810 | 0.755-0.863 | 0.620 |
| Z-DNA Hunter model 2 size 10 score 60% | 0.616 | 0.986 | 0.800 | 0.769-0.828 | 0.600 |
| Z-DNA Hunter model 2 | 0.571 | 1.000 | 0.790 | 0.757-0.811 | 0.570 |
| Z-DNA Hunter model 1 | 0.564 | 1.000 | 0.780 | 0.755-0.809 | 0.560 |
| Z-DNA Hunter model 2 size 8 score 65% | 0.589 | 0.863 | 0.730 | 0.676-0.773 | 0.450 |
| E7 priority >= A_validovat | 0.369 | 0.986 | 0.677 | 0.646-0.707 | 0.355 |
| ZDNABERT HG18 ChIP-seq | 0.877 | 0.384 | 0.630 | 0.573-0.693 | 0.260 |
| E7 priority >= C_manualna_kuracia | 1.000 | 0.014 | 0.507 | 0.500-0.522 | 0.014 |

## Top kandidati najlepsieho modelu

| Poradie | Kandidat | Lokacia | Priorita | Expert | Fronta |
|---:|---|---|---:|---:|---|
| 1 | ZFR_0127 | generic_eukaryote chr1:689586-689926 | 96.1 | 100.0 | A_validovat |
| 2 | ZFR_0311 | generic_eukaryote chr5:1576347-1576670 | 93.7 | 90.4 | A_validovat |
| 3 | ZFR_0041 | generic_eukaryote chr5:272839-272904 | 93.1 | 96.7 | A_validovat |
| 4 | ZFR_0018 | generic_eukaryote chrX:163620-163711 | 92.4 | 100.0 | A_validovat |
| 5 | ZFR_0105 | generic_eukaryote chr3:581280-581411 | 91.9 | 96.3 | A_validovat |
| 6 | ZFR_0207 | generic_eukaryote chr3:1074600-1074957 | 91.7 | 94.6 | A_validovat |
| 7 | ZFR_0319 | generic_eukaryote chr1:1613890-1613974 | 91.4 | 91.3 | A_validovat |
| 8 | ZFR_0225 | generic_eukaryote chr3:1159665-1159682 | 90.9 | 98.5 | A_validovat |
| 9 | ZFR_0010 | generic_eukaryote chr4:125090-125464 | 90.3 | 95.6 | A_validovat |
| 10 | ZFR_0273 | generic_eukaryote chr3:1393110-1393175 | 90.1 | 95.8 | A_validovat |
| 11 | ZFR_0185 | generic_eukaryote chr5:967520-967724 | 89.9 | 91.0 | A_validovat |
| 12 | ZFR_0146 | generic_eukaryote chr2:779756-779995 | 89.7 | 87.2 | A_validovat |
| 13 | ZFR_0239 | generic_eukaryote chr5:1228660-1229015 | 89.4 | 100.0 | A_validovat |
| 14 | ZFR_0305 | generic_eukaryote chr5:1545217-1545285 | 89.4 | 98.7 | A_validovat |
| 15 | ZFR_0097 | generic_eukaryote chr1:543316-543709 | 89.0 | 97.8 | A_validovat |
