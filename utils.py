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
