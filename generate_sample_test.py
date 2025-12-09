import pandas as pd


inp_path = "data/processed/en_test_2k4.json"
inp_path = "data/processed/vi_test_2k1.json"

test = pd.read_json(inp_path)
max_samples = int(test.function.value_counts().max())

max_each_fn = 10
max_each_fn = min(max_each_fn, max_samples)


output_path = f"data/en_test_each_max_{max_each_fn}.json"
output_path = f"data/vi_test_each_max_{max_each_fn}.json"


import pandas as pd

# df is your dataframe
# Columns: user_message, tool_calls, function
import pandas as pd
samples: pd.DataFrame = (
    test.groupby("function")
      .apply(lambda g: g.sample(n=min(len(g), max_each_fn), random_state=42))
      .reset_index(drop=True)
)

print("Test set includes {} samples".format(len(samples)))

samples.to_json(
  output_path,
  orient="records",
  force_ascii=False,
  indent=4
)