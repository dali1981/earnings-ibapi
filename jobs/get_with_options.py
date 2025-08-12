from ibx_repos.contract_descriptions import ContractDescriptionsRepository
import pandas as pd

repo = ContractDescriptionsRepository("data/contract_descriptions")

# Pull the columns we need
df = repo.read(columns=["symbol","query","conid","derivative_sec_types","as_of_date","sec_type"])

# Keep latest row per (query, conid) to avoid duplicates across days
latest_idx = (df.groupby(["conid"])["as_of_date"].idxmax()
                if len(df) else pd.Index([]))
latest = df.loc[latest_idx] if len(latest_idx) else df

# Filter rows where derivative_sec_types includes "OPT"
has_opt = latest["derivative_sec_types"].apply(lambda xs: "OPT" in xs)

# Result: unique symbols (optionally restrict to equities)
res = (latest[has_opt]
       .query("sec_type == 'STK'")
       .drop_duplicates(subset=["symbol"])
       .sort_values("symbol"))

# print(res[["symbol","conid","query","as_of_date"]].to_string(index=False))

for s in ["DEC", "CMCL", "NIU"]:
    print(f"{s} : {s in list(res['symbol'])}")
