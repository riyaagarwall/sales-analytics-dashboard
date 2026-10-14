import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)


def load_data():
    df = pd.read_csv("Data/sales.csv")
    df = df.drop_duplicates()
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    return df[["Order ID", "Order Date", "Region", "Category", "Sub-Category", "Segment", "Sales"]]


df = load_data()


def monthly_series():
    return df.set_index("Order Date")["Sales"].resample("MS").sum()


def make_features(dates, t):
    """Column 0 = month number (trend). Columns 1-11 = Feb..Dec flags (January is the baseline)."""
    X = np.zeros((len(dates), 12))
    X[:, 0] = t
    for i, d in enumerate(dates):
        if d.month > 1:
            X[i, d.month - 1] = 1
    return X


def holdout_mae(X, y, test):
    """Train without the last `test` months, then measure the error on them."""
    model = LinearRegression().fit(X[:-test], y[:-test])
    return round(float(mean_absolute_error(y[-test:], model.predict(X[-test:]))), 2)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/summary")
def summary():
    orders = df["Order ID"].nunique()
    total = float(df["Sales"].sum())
    return jsonify(
        total_sales=round(total, 2),
        orders=orders,
        avg_order_value=round(total / orders, 2),
        top_category=df.groupby("Category")["Sales"].sum().idxmax(),
    )


@app.route("/api/sales/monthly")
def monthly():
    m = monthly_series().round(2)
    return jsonify(labels=[d.strftime("%b %Y") for d in m.index], values=m.tolist())


@app.route("/api/sales/by-category")
def by_category():
    g = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).round(2)
    return jsonify(labels=g.index.tolist(), values=g.tolist())


@app.route("/api/sales/by-region")
def by_region():
    g = df.groupby("Region")["Sales"].sum().sort_values(ascending=False).round(2)
    return jsonify(labels=g.index.tolist(), values=g.tolist())


@app.route("/api/forecast")
def forecast():
    m = monthly_series()
    y = m.values
    t = np.arange(len(m))
    horizon, test = 12, 12

    future_dates = pd.date_range(m.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future_t = np.arange(len(m), len(m) + horizon)

    # Model 1: straight line (trend only)
    X_lin = t.reshape(-1, 1)
    lin = LinearRegression().fit(X_lin, y)

    # Model 2: trend + month-of-year seasonality
    X_sea = make_features(m.index, t)
    sea = LinearRegression().fit(X_sea, y)

    return jsonify(
        labels=[d.strftime("%b %Y") for d in future_dates],
        linear=[round(float(v), 2) for v in lin.predict(future_t.reshape(-1, 1))],
        seasonal=[round(float(v), 2) for v in sea.predict(make_features(future_dates, future_t))],
        metrics={
            "linear": {"mae": holdout_mae(X_lin, y, test), "r2": round(float(lin.score(X_lin, y)), 3)},
            "seasonal": {"mae": holdout_mae(X_sea, y, test), "r2": round(float(sea.score(X_sea, y)), 3)},
        },
    )


if __name__ == "__main__":
    app.run(debug=True)