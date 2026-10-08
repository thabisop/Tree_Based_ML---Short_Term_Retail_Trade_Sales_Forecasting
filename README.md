# Tree-Based Machine Learning for Short-Term Retail Trade Sales Forecasting in South Africa

This repository contains the code and data pipeline for a study that compares
tree-based machine-learning models with classical time-series models for
one-to-five-month-ahead forecasting of aggregate South African retail trade
sales (constant 2019 prices, January 1986 to May 2026). The work accompanies
the SACAIR 2026 conference paper of the same title.

The models compared are a naive forecast, a seasonal-naive forecast, ARIMA,
SARIMA (SARIMAX without exogenous regressors), SARIMAX, Random Forest, XGBoost
and LightGBM. The evaluation is leakage-controlled and uses both a fixed
January-to-May-2026 holdout and a rolling-origin walk-forward with 307 origins.
Forecasts are compared with modified Diebold-Mariano tests under
Benjamini-Hochberg control, uncertainty is quantified with split-conformal
prediction intervals, and the tree models are explained with SHAP.

The project runs as a three-stage pipeline:

1. **Data aggregation** builds the modelling dataset from raw primary sources
   (run locally in Anaconda with Jupyter Notebook).
2. **Exploratory data analysis (EDA)** profiles the series and informs the
   preprocessing and feature decisions (run in Google Colab).
3. **Modelling** trains and evaluates every model and produces the paper's
   tables and figures (run in Google Colab).

---

## 1. Repository contents

```text
.
├── README.md                                 (this document)
├── requirements.txt                          Python package list (mainly for the local data-aggregation stage)
├── DataAggregation.ipynb                     Stage 1: builds final_model_dataset_corrected.csv from raw sources (Anaconda / Jupyter)
├── EDA_RetailTradeSales.ipynb                Stage 2: exploratory data analysis (Google Colab)
└── Modelling_Script.ipynb                    Stage 3: model training, evaluation and figures (Google Colab)
```

The raw source files (Stats SA, BIS and BER exports) and the generated dataset
are **not** stored in the repository. They are described in Section 2 and are
placed alongside the data-aggregation notebook at run time.

---

## 2. Stage 1: Data aggregation (Anaconda + Jupyter Notebook, local)

### 2.1 What it does

`DataAggregation.ipynb` rebuilds the full monthly modelling dataset
(1986 to 2026) end-to-end from raw primary sources, with no pre-amalgamated
input. It constructs the retail target, aligns every macroeconomic covariate,
applies publication lags, and writes a single CSV used by the EDA and modelling
notebooks.

The retail target is the Stats SA P6242.1 Total series (constant prices, not
seasonally adjusted), growth-spliced across the 1995-to-2019 base change so the
base seam is removed while each era's real dynamics are preserved. The script
validates the target against the official P6242.1 (May 2026) release:
December 2025 = 138,834 and January-to-May 2026 =
97,326 / 95,209 / 99,422 / 96,677 / 101,008 (R million, constant 2019 prices).

### 2.2 Raw sources and required files

Download the following from their official portals and place **all of them in
the same folder as the data-aggregation notebook** (the notebook reads them by
file name from its working directory).

| Variable | Source | Series / publication | Files expected in the working folder |
|---|---|---|---|
| Retail trade sales (target) | Statistics South Africa (statssa.gov.za) | P6242.1, Total, constant prices, NSA | `Excel 1968 to 1980.xls`, `Excel 1981 to 1990.xls`, `Excel 1991 to 2000.xls`, `Excel from 2001.xls`, `Retail trade sales from 2002.xlsx` |
| Electricity production index | Statistics South Africa (statssa.gov.za) | P4141, base 2019 = 100, NSA | `Excel table from 1985 to 1989_Electricity.xlsx`, `Excel table from 1990 to 1999_Electricity.xlsx`, `Excel table from 2000_Electricity.xlsx` |
| CPI (consumer price index) | BIS Data Portal (data.bis.org) | series `M.ZA.628` (index 2010 = 100) | `bis_dp_search_export_CPI.xlsx` |
| Exchange rate (USD/ZAR) | BIS Data Portal (data.bis.org) | series `M.ZA.ZAR.A` (monthly, PERIOD AVERAGE) | `bis_dp_search_export_exchangeRateUSDZAR_average.xlsx` |
| Consumer confidence (CCI) | Bureau for Economic Research (ber.ac.za) | FNB/BER Consumer Confidence Index (quarterly) | `FNB_BER_CCI.xlsx` |

Download links (as cited in the paper's references):

- CPI, BIS series `M.ZA.628`:
  https://data.bis.org/topics/CPI/BIS,WS_LONG_CPI,1.0/M.ZA.628
- Exchange rate, BIS series `M.ZA.ZAR.A` (period average):
  https://data.bis.org/topics/XRU/BIS,WS_XRU,1.0/M.ZA.ZAR.A
- FNB/BER Consumer Confidence Index (2026 history data), BER:
  https://www.ber.ac.za/Documents/Info/fnbber-consumer-confidence-index-2026-history-data?doc=ea03af5e-5e73-4155-aebe-cfc7734afcfc
- Retail trade sales, Statistics South Africa statistical release P6242.1:
  https://www.statssa.gov.za/?page_id=1854&PPN=P6242.1&SCH=74315
- Electricity production index, Statistics South Africa statistical release P4141:
  https://www.statssa.gov.za/?page_id=1854&PPN=P4141&SCH=74349

Notes:

- The legacy retail vintages are `.xls` files and are read with the `xlrd`
  engine, so `xlrd` must be installed (it is in `requirements.txt`).
- The exchange rate must be the **period-average** export (`M.ZA.ZAR.A`), not
  the end-of-period one. If you re-export it, keep the file name above or update
  `FX_FILE` in the notebook.
- The quarterly CCI is carried forward to monthly by a step function inside the
  notebook, so only the quarterly workbook is needed.

### 2.3 How to run

1. Create and activate a conda environment, then install the requirements:

   ```bash
   conda create -n retail-forecasting python=3.11 -y
   conda activate retail-forecasting
   pip install -r requirements.txt
   ```

2. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook
   ```

   then open `DataAggregation.ipynb`.

3. Confirm the raw files from Section 2.2 are in the same folder as the
   notebook, then run all cells (Kernel then Restart & Run All).

4. Output: the notebook writes

   ```text
   final_model_dataset_corrected.csv
   ```

   a semicolon-separated (`sep=";"`) monthly dataset spanning January 1986 to
   May 2026, with the target, the macro covariates, lagged and rolling features,
   and calendar indicators. This CSV is the single input for Stages 2 and 3.

---

## 3. Stage 2: Exploratory data analysis (Google Colab)

### 3.1 What it does

The EDA notebook profiles `final_model_dataset_corrected.csv`: trend and
seasonality, the December effect, stationarity checks (for example the
Augmented Dickey-Fuller test), autocorrelation structure, and the screening of
the macroeconomic covariates. These results motivate the preprocessing and
feature choices used in the modelling stage.

### 3.2 How to run

1. Upload `final_model_dataset_corrected.csv` (from Stage 1) to your Google
   Drive, for example to a folder named `MIT807 Project`.
2. Open the EDA notebook in Google Colab.
3. Run the first cells to mount Google Drive, then point the data path at the
   CSV you uploaded (the notebooks read it with `pd.read_csv(path, sep=";")`).
4. Run all cells (Runtime then Run all).

No GPU is required for the EDA.

---

## 4. Stage 3: Modelling (Google Colab)

### 4.1 What it does

The modelling notebook trains and evaluates every model on the dataset from
Stage 1 and produces the paper's results: the fixed-holdout comparison, the
rolling-origin walk-forward, the Diebold-Mariano tests with Benjamini-Hochberg
control, the leave-one-group-out ablations, the split-conformal prediction
intervals, and the SHAP explanations.

### 4.2 Setup and how to run

1. Upload `final_model_dataset_corrected.csv` to Google Drive. By default the
   notebook reads it from:

   ```text
   /content/drive/MyDrive/MIT807 Project/final_model_dataset_corrected.csv
   ```

   and writes its outputs to:

   ```text
   /content/drive/MyDrive/MIT807 Project/outputs
   ```

   If you use a different folder, update `DATA_PATH` and `OUTPUT_DIR` in the
   first configuration cell. The notebook falls back to a local
   `final_model_dataset_corrected.csv` in the working directory if the Drive
   path is not found.

2. Open the modelling notebook in Google Colab and run all cells
   (Runtime then Run all).

3. Outputs written to the output folder include the results tables as CSV files
   (`holdout_results.csv`, `wf_results.csv`, `dm_benjamini_hochberg.csv`,
   `ablation_results.csv`, `sarimax_exog_ablation.csv`,
   `conformal_rolling_coverage.csv`, `conformal_rolling_widths.csv`,
   `conformal_holdout_intervals.csv`, and the best-tree conformal CSVs) and the
   figures (`plot_model_comparison.png`, `plot_insSample_Forecast.png`,
   `fig4a_shap_beeswarm.png`, `fig4b_shap_waterfall.png`, and the companion
   plots). A version-print cell reports the exact library versions for the
   reproducibility statement.

### 4.3 Runtime

This pipeline is CPU-bound. The classical models use statsmodels and the tree
models run on CPU as configured, so a **standard or high-RAM CPU runtime is
sufficient**; a GPU is not required. The single heaviest cell is the SARIMAX
exogenous-variable ablation, which refits SARIMAX on every rolling origin and
can take roughly 20 to 45 minutes; the rest of the notebook is considerably
faster.

---

## 5. Requirements and environments

Two environments are used:

- **Stage 1 (data aggregation)** runs locally in Anaconda with Jupyter Notebook.
  Install the packages in `requirements.txt` into a conda environment as shown
  in Section 2.3.
- **Stages 2 and 3 (EDA and modelling)** run in Google Colab, which already
  provides most of the packages. The modelling notebook was run on Colab with
  Python 3.13.15 and the library versions pinned in `requirements.txt`
  (statsmodels 0.15.0, xgboost 3.4.1, lightgbm 4.6.0, scikit-learn 1.6.1). Those
  pins are the versions reported in the paper; install them in Colab only if you
  need to reproduce the exact environment.

---

## 6. Reproducibility notes

- The dataset is reproducible end-to-end from the primary sources in Section 2;
  no pre-amalgamated covariate file is used.
- The retail target matches the official Stats SA P6242.1 (May 2026) release at
  the validation checkpoints listed in Section 2.1.
- The modelling notebook fixes a single random seed (42) and holds all model
  hyperparameters constant across origins and horizons, so re-running it on the
  same dataset reproduces the reported tables and figures.

---

## 7. Contact

For questions or reproducibility issues:

**Thabiso Msimango**
Personal: tmsimango523@gmail.com
University of Pretoria: u25738497@tuks.co.za
