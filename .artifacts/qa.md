# axiom clean run

`clean_config_hash` **98d62d99d8f6**

27,905 segments · 38,758,930 usable bars · 27,508,145 context-512 windows

## Usable corpus

| Source | Asset class | Market | Freq | Series | Segments | Usable bars | Windows @512 | Longest |
|---|---|---|---|---:|---:|---:|---:|---:|
| binance_vision | crypto | spot | 1d | 200 | 203 | 307,462 | 204,877 | 3,290 |
| binance_vision | crypto | spot | 1h | 200 | 2,401 | 7,050,386 | 5,908,383 | 29,866 |
| binance_vision | crypto | um | 1d | 100 | 115 | 126,915 | 70,398 | 2,423 |
| binance_vision | crypto | um | 1h | 100 | 211 | 3,045,256 | 2,940,249 | 58,152 |
| dukascopy | commodity | commodity | 1d | 6 | 76 | 21,946 | 2,824 | 3,335 |
| dukascopy | commodity | commodity | 1h | 6 | 475 | 345,487 | 131,686 | 38,825 |
| dukascopy | fx | fx | 1d | 20 | 87 | 138,915 | 96,440 | 4,156 |
| dukascopy | fx | fx | 1h | 20 | 640 | 2,674,842 | 2,362,617 | 51,405 |
| stooq | equity | us | 1d | 9995 | 23,697 | 25,047,721 | 15,790,671 | 6,729 |

## Drop rates by rule

| Source | Freq | Rule | Bars dropped | Runs excised | Segments dropped | % of slice |
|---|---|---|---:|---:|---:|---:|
| binance_vision | 1d | gap | 0 | 0 | 0 | 0.000% |
| binance_vision | 1d | illiquid | 757 | 6 | 0 | 0.174% |
| binance_vision | 1d | jump | 0 | 0 | 0 | 0.000% |
| binance_vision | 1d | min_length | 494 | 0 | 16 | 0.113% |
| binance_vision | 1d | session_filter | 0 | 0 | 0 | 0.000% |
| binance_vision | 1d | stagnant | 16 | 4 | 0 | 0.004% |
| binance_vision | 1h | gap | 0 | 0 | 0 | 0.000% |
| binance_vision | 1h | illiquid | 18,275 | 12 | 0 | 0.175% |
| binance_vision | 1h | jump | 0 | 0 | 0 | 0.000% |
| binance_vision | 1h | min_length | 307,333 | 0 | 4,916 | 2.941% |
| binance_vision | 1h | session_filter | 0 | 0 | 0 | 0.000% |
| binance_vision | 1h | stagnant | 28,265 | 6,301 | 0 | 0.271% |
| dukascopy | 1d | gap | 0 | 0 | 0 | 0.000% |
| dukascopy | 1d | illiquid | 1,439 | 167 | 0 | 0.852% |
| dukascopy | 1d | jump | 0 | 0 | 0 | 0.000% |
| dukascopy | 1d | min_length | 6,611 | 0 | 260 | 3.914% |
| dukascopy | 1d | session_filter | 0 | 0 | 0 | 0.000% |
| dukascopy | 1d | stagnant | 8 | 2 | 0 | 0.005% |
| dukascopy | 1h | gap | 0 | 0 | 0 | 0.000% |
| dukascopy | 1h | illiquid | 75,121 | 9,717 | 1 | 2.154% |
| dukascopy | 1h | jump | 0 | 0 | 0 | 0.000% |
| dukascopy | 1h | min_length | 187,826 | 0 | 10,729 | 5.387% |
| dukascopy | 1h | session_filter | 202,633 | 0 | 0 | 5.811% |
| dukascopy | 1h | stagnant | 1,023 | 212 | 0 | 0.029% |
| stooq | 1d | gap | 0 | 0 | 0 | 0.000% |
| stooq | 1d | illiquid | 15,822 | 2,584 | 464 | 0.057% |
| stooq | 1d | jump | 0 | 0 | 0 | 0.000% |
| stooq | 1d | min_length | 2,620,914 | 0 | 306,837 | 9.439% |
| stooq | 1d | session_filter | 0 | 0 | 0 | 0.000% |
| stooq | 1d | stagnant | 82,777 | 16,587 | 104 | 0.298% |

## Red flags

| Check | Subject | Value | Limit | Detail |
|---|---|---:|---:|---|
| major_series_loss | `raw/dukascopy/fx/1h/USDJPY.parquet` | 7.364 | 1.0 | USDJPY lost 7.364% of its bars |
| major_series_loss | `raw/dukascopy/commodity/1d/XAUUSD.parquet` | 6.253 | 1.0 | XAUUSD lost 6.253% of its bars |
| major_series_loss | `raw/dukascopy/fx/1h/EURUSD.parquet` | 5.991 | 1.0 | EURUSD lost 5.991% of its bars |
| major_series_loss | `raw/dukascopy/commodity/1h/XAUUSD.parquet` | 5.63 | 1.0 | XAUUSD lost 5.63% of its bars |

**Every row above needs an investigation written against it before the v0.3 gate.**

## Top 20 most-cut series

| Artifact | Total | Kept | Dropped | % | By rule |
|---|---:|---:|---:|---:|---|
| `raw/stooq/us/1d/B/BHV.parquet` | 4,782 | 0 | 4,782 | 100.00% | min_length: 4,778, stagnant: 4 |
| `raw/stooq/us/1d/G/GRF.parquet` | 4,547 | 0 | 4,547 | 100.00% | min_length: 4,503, stagnant: 44 |
| `raw/stooq/us/1d/K/KFFB.parquet` | 4,309 | 0 | 4,309 | 100.00% | min_length: 4,276, stagnant: 33 |
| `raw/stooq/us/1d/N/NEN.parquet` | 3,794 | 0 | 3,794 | 100.00% | min_length: 3,778, stagnant: 16 |
| `raw/stooq/us/1d/C/CKX.parquet` | 3,608 | 0 | 3,608 | 100.00% | min_length: 3,564, stagnant: 44 |
| `raw/stooq/us/1d/M/MGYR.parquet` | 3,606 | 0 | 3,606 | 100.00% | illiquid: 4, min_length: 3,581, stagnant: 21 |
| `raw/stooq/us/1d/K/KTN.parquet` | 3,554 | 0 | 3,554 | 100.00% | min_length: 3,550, stagnant: 4 |
| `raw/stooq/us/1d/L/LXP_C.parquet` | 3,504 | 0 | 3,504 | 100.00% | min_length: 3,496, stagnant: 8 |
| `raw/stooq/us/1d/P/PBHC.parquet` | 3,377 | 0 | 3,377 | 100.00% | illiquid: 10, min_length: 3,321, stagnant: 46 |
| `raw/stooq/us/1d/S/SPG_J.parquet` | 3,331 | 0 | 3,331 | 100.00% | min_length: 3,319, stagnant: 12 |
| `raw/stooq/us/1d/P/PCG_D.parquet` | 3,303 | 0 | 3,303 | 100.00% | min_length: 3,283, stagnant: 20 |
| `raw/stooq/us/1d/C/CMS_B.parquet` | 3,237 | 0 | 3,237 | 100.00% | min_length: 3,233, stagnant: 4 |
| `raw/stooq/us/1d/K/KTH.parquet` | 3,174 | 0 | 3,174 | 100.00% | min_length: 3,154, stagnant: 20 |
| `raw/stooq/us/1d/I/IOR.parquet` | 3,160 | 0 | 3,160 | 100.00% | min_length: 3,144, stagnant: 16 |
| `raw/stooq/us/1d/P/PCG_B.parquet` | 2,988 | 0 | 2,988 | 100.00% | min_length: 2,980, stagnant: 8 |
| `raw/stooq/us/1d/C/CTA_A.parquet` | 2,971 | 0 | 2,971 | 100.00% | min_length: 2,955, stagnant: 16 |
| `raw/stooq/us/1d/H/HFBL.parquet` | 2,948 | 0 | 2,948 | 100.00% | min_length: 2,886, stagnant: 62 |
| `raw/stooq/us/1d/G/GTN-A.parquet` | 2,926 | 0 | 2,926 | 100.00% | min_length: 2,926 |
| `raw/stooq/us/1d/I/IROQ.parquet` | 2,836 | 0 | 2,836 | 100.00% | min_length: 2,798, stagnant: 38 |
| `raw/stooq/us/1d/I/IPB.parquet` | 2,778 | 0 | 2,778 | 100.00% | min_length: 2,768, stagnant: 10 |

---

Built by `axiom clean report` from `clean/v1/segments.parquet` and `clean/v1/dropstats.parquet`. Raw bars are unchanged; cleaning produced only this metadata (ADR-0018).
