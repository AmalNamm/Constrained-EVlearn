import os
import pandas as pd

# Step 1: Reading the Files
folder_path = 'C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn\\TiagoData\\marginalpdbcpt_2022'
all_data = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    with open(filepath, 'r') as file:
        # Skip the header line
        next(file)
        # Read and parse the content
        for line in file:
            if line.strip() != '*':
                all_data.append(line.strip().split(';'))

# Step 2: Parsing the Data
# Convert the list of lists into a DataFrame
columns = ['Year', 'Month', 'Day', 'Hour', 'Electricity Pricing [$]', 'Electricity Pricing 2 [$]', "Nada"]
df = pd.DataFrame(all_data, columns=columns)

# We only need the 'Electricity Pricing [$]' column for further calculations, so let's keep that
df['Electricity Pricing [$]'] = df['Electricity Pricing [$]'].astype(float)

# Step 3: Dividing the Electricity Prices by 1000
df['Electricity Pricing [$]'] = df['Electricity Pricing [$]'] / 1000

# Step 3: Generating Predictions
df['6h Prediction Electricity Pricing [$]'] = df['Electricity Pricing [$]'].shift(-6)
df['12h Prediction Electricity Pricing [$]'] = df['Electricity Pricing [$]'].shift(-12)
df['24h Prediction Electricity Pricing [$]'] = df['Electricity Pricing [$]'].shift(-24)


# Select the first 24*7 lines
first_rows = df.iloc[:5086]

# Drop the first 24*7 lines from the original dataframe
df = df.iloc[5086:]

# Append the first 24*7 lines to the end
df = df.append(first_rows, ignore_index=True)

print(df)

# Save the final DataFrame to a new CSV file
df.to_csv('electricity_pricing_predictions_test.csv', index=False)

# Step 4: Dropping Unnecessary Columns
# Specify the columns you want to drop
columns_to_drop = ['Year','Month','Day','Nada', 'Electricity Pricing 2 [$]', 'Hour']
df = df.drop(columns=columns_to_drop)







# Save the final DataFrame to a new CSV file
df.to_csv('electricity_pricing_predictions.csv', index=False)

# Print the result
print(df.head())
