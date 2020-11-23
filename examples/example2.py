import pandas as pd
from reshape_tools.make_recurrent import make_recurrent
from sample_data.make_sample_data import sample_data1, sample_data2


def main():
    df = sample_data1()
    df["month"] = pd.to_datetime(df["times"]).dt.to_period("M")
    arr = make_recurrent(
        df,
        3,
        "times",
        partition_by="month",
        drop_order_by=False,
        drop_partition_by=False,
    )
    print(df)
    print("***********************************************")
    print(arr)
    print(f"Output shape is {arr.shape}")


if __name__ == "__main__":
    main()
