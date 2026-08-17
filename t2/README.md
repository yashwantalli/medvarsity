# NumPy, Pandas & PyTorch Practice Notebooks

A set of hands-on exercises covering core data manipulation with NumPy/Pandas (Titanic dataset) and building neural network models with PyTorch (Apple stock price prediction).

## Contents

| Notebook | Dataset | Focus |
|---|---|---|
| `t2_numpy.ipynb` | Titanic (`titanic/test.csv`) | NumPy array operations, statistics, DataFrame indexing/filtering |
| `t2_pandas.ipynb` | Titanic (`titanic/train.csv`) | Data cleaning, missing values, column ops, feature engineering |
| `t2_pytorch.ipynb` | Apple stock (`apple_stock/AAPL.csv`) | Tensors, DataLoaders, Linear Regression and LSTM for time-series prediction |

## `t2_numpy.ipynb` — NumPy Fundamentals

Works with the Titanic test set to practice array and DataFrame operations:

- Converting a DataFrame to a NumPy array (`.to_numpy()`)
- Descriptive statistics on `Age` (mean, median, std, min, max) after dropping nulls
- Conditional row selection with `.loc` (e.g., youngest/oldest passengers, age > 40)
- Column enumeration and positional slicing with `.iloc`

**Sample findings:** mean passenger age ≈ 30.3, median 27.0, std ≈ 14.2, range 0.17–76.0 years.

## `t2_pandas.ipynb` — Pandas Data Cleaning

Works with the Titanic training set to practice a typical cleaning workflow:

- Initial inspection: `.head()`, `.shape`, `.info()`, null counts (`Age`: 177 missing, `Cabin`: 687 missing, `Embarked`: 2 missing), duplicate check
- Handling missing data three ways: `dropna()`, `fillna(0)`, and median imputation for `Age`
- Renaming columns (`PassengerId` → `PId`, etc.)
- Dropping columns (`Embarked`, `SibSp`, `Parch`)
- Feature engineering: bucketing `Age` into `Age_Group` (Child / Adult / Senior) via `.apply()`
- Type casting `Age` to `int` after imputation

## `t2_pytorch.ipynb` — PyTorch for Time-Series Prediction

Builds up from a plain linear model to an LSTM for predicting Apple stock's daily high price from a rolling window of past prices.

**Pipeline:**
1. Load and inspect `AAPL.csv` (10,468 rows), parse `Date` to datetime, confirm no missing values
2. Convert the `High` price series into supervised-learning windows (window size = 50) with a custom `seq()` function
3. Wrap data in `torch.Tensor`, `TensorDataset`, and `DataLoader` (batch size 64)
4. **Baseline:** single `nn.Linear(50, 1)` layer trained with `Adam` (lr=0.001) and `MSELoss`, first on raw prices, then repeated with `MinMaxScaler`-normalized prices — scaling drops the loss from ~2.2 to ~0.0001, plotted on a log scale for comparison
5. **80/20 train/test split** on the scaled series, rebuilt into windowed tensors (`x_train`: 8,324×50×1, `x_test`: 2,044×50×1)
6. **LSTM model** (`StockLSTM`): a single `nn.LSTM(input_size=1, hidden_size=64, batch_first=True)` layer feeding a final `nn.Linear(64, 1)` head, trained for 50 epochs (loss converges from ~0.0095 to ~0.000026)
7. Evaluation and visualization comparing predicted vs. actual prices for the linear and LSTM models

## Requirements

```
numpy
pandas
torch
scikit-learn
matplotlib
```

## Data

- `titanic/train.csv`, `titanic/test.csv` — from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic)
- `apple_stock/AAPL.csv` — historical daily OHLCV data for Apple (AAPL) stock

Place these under a local `titanic/` and `apple_stock/` folder relative to the notebooks before running.

## Notes

These are learning/practice notebooks rather than a polished pipeline — variable naming and structure reflect iterative experimentation (e.g., multiple `df_dupN` variants in the pandas notebook to compare cleaning strategies).
