import pandas as pd


def create_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date_arrival"] = (
        pd.to_datetime(df["date_arrival"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    df = df[df["receival_status"] == "Completed"]
    df = df[["rm_id", "date_arrival", "net_weight"]]

    # drop rows with missing rm_id
    df = df.dropna(subset=["rm_id"])
    df["rm_id"] = df["rm_id"].astype(int)
    rm_ids = df["rm_id"].unique().tolist()

    df = (
        df.groupby(["rm_id", "date_arrival"], as_index=False)
        .agg(
            net_weight=("net_weight", "sum"),
            # num_arrivals=('net_weight', 'count')   # Anzahl der Zeilen / Receival Items
        )
        .sort_values(["rm_id", "date_arrival"])
    )

    # df["cum_net_weight"] = df.groupby("rm_id")["net_weight"].cumsum()

    start_date = df["date_arrival"].min()
    end_date = pd.Timestamp("2024-12-31")
    lst = []
    for name, group in df.groupby("rm_id"):
        full_index = pd.date_range(start_date, end_date, freq="D")
        group = (
            group.set_index("date_arrival")
            .reindex(full_index)
            .rename_axis("date_arrival")
            .reset_index()
        )
        group["rm_id"] = name
        group["net_weight"] = group["net_weight"].fillna(0)
        # group["cum_net_weight"] = group["cum_net_weight"].fillna(method="ffill")

        lst.append(group)

    df = pd.concat(lst)
    return df



# input dataframe : contains all rm_id, net_weight_sum and date (at least these three columns)
# name of the parameter : df_final
def print_submission(df_final, filename="submission.csv"):

    df_final["rm_id"] = pd.to_numeric(df_final["rm_id"], errors="coerce").astype("Int64")

    # 3) Load the ID mapping
    ids = pd.read_csv("./data/prediction_mapping.csv")  # or your actual path
    ids["forecast_start_date"] = pd.to_datetime(ids["forecast_start_date"])
    ids["forecast_end_date"]   = pd.to_datetime(ids["forecast_end_date"])
    ids["rm_id"]    = pd.to_numeric(ids["rm_id"], errors="coerce").astype("Int64")
    ids = ids.rename(columns={
        "forecast_end_date": "date"
    })

    # 4) For each ID, get cum at end_date (<= end_date)

    out = (
        ids.merge(df_final, on=["rm_id", "date"], how="left")
        .assign(cum=lambda d: d["cum"].fillna(0))   # cum=0 quand absent
        .sort_values(["rm_id", "date"])
        .reset_index(drop=True)
    )


    # 5) Cum at day before start_date
    submission = (
        out[["ID", "cum"]]
        .rename(columns={"cum": "predicted_weight"})   # if your file needs "predicted_weight"
        .fillna({"predicted_weight": 0})
        .sort_values("ID")
    )

    submission.to_csv("./submissions/" + filename, index=False)
    print(submission.head())
