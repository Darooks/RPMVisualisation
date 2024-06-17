import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 4000
TOLERANCE = 20

df = pd.read_parquet('shaft_speed.parquet')

df_avg = df['shaft_speed'].rolling(window=WINDOW_SIZE).mean()
df_std = df['shaft_speed'].rolling(window=WINDOW_SIZE).std()

df_faulty = pd.DataFrame(columns=df.columns)

# <First idea> to find the faulty and correct samples
# df_faulty = df[faulty_mask]
# df_correct = df[correct_mask]
# for idx, row in df.iterrows():
#     if np.isnan(df_avg[idx]) or np.isnan(df_std[idx]):
#         continue
#
#     if abs(row['shaft_speed'] - df_avg[idx]) > df_std[idx]:
#         df_faulty.append(row)
#     else:
#         df_correct.append(row)
# </First idea>


# But I can use masks and it is way more efficient - yes, i googled it :)
# Find faulty and correct rows using vectorized operations
faulty_mask = np.abs(df['shaft_speed'] - df_avg) > df_std + TOLERANCE
correct_mask = ~faulty_mask

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(df.index, df['shaft_speed'], label='All Data', color='blue')
plt.plot(df.index[faulty_mask], df['shaft_speed'][faulty_mask], 'ro', label='Faulty Data')
plt.plot(df.index[correct_mask], df['shaft_speed'][correct_mask], color='orange', label='Correct Data')

plt.xlabel('Time')
plt.ylabel('RPM')
plt.title('Shaft Speed Data')
plt.legend()
plt.grid(True)
plt.show()
