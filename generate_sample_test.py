import pandas as pd



test = pd.read_json("data/english_2k4_vivi_global_function_calling_test_dataset.json")


import pandas as pd

# df is your dataframe
# Columns: user_message, tool_calls, function
min_each_fn = 1

samples = (
    test.groupby("function")
      .apply(lambda g: g.sample(n=min(len(g), min_each_fn), random_state=42))
      .reset_index(drop=True)
)



samples.to_json(f"data/test_each_{min_each_fn}.json", orient="records")