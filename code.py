import pandas as pd
import numpy as np
import os
import zipfile

print("Available files:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"{filepath}: {size_mb:.1f} MB")

# Let's see the actual structure of your input folder
print("Full directory structure:")
for root, dirs, files in os.walk('/kaggle/input'):
    level = root.replace('/kaggle/input', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        size_mb = os.path.getsize(os.path.join(root, file)) / (1024*1024)
        print(f'{subindent}{file} ({size_mb:.1f} MB)')

print("=== FULL DATASET INFO ===")
total_rows = sum(1 for line in open('/kaggle/input/usa-real-estate/realtor-data.zip.csv')) - 1
print(f"Total rows: {total_rows:,}")

df_sample = pd.read_csv('/kaggle/input/usa-real-estate/realtor-data.zip.csv', nrows=1000)
print(f"Shape of sample: {df_sample.shape}")

print(f"\nCOLUMN BREAKDOWN:")
for i, col in enumerate(df_sample.columns, 1):
    print(f"{i:2}. {col}")

print(f"\nDATA TYPES:")
print(df_sample.dtypes)

print(f"\nFIRST 3 ACTUAL ROWS:")
print(df_sample.head(3))

print("CURRENT MEMORY USAGE:")
print(f"Sample memory: {df_sample.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"Estimated full dataset: {(df_sample.memory_usage(deep=True).sum() * 2226382 / 1000) / 1024**2:.1f} MB")

print("\nMISSING VALUES:")
print(df_sample.isnull().sum())

print("\nUNIQUE VALUES (sample):")
for col in ['status', 'state']:
    print(f"{col}: {df_sample[col].value_counts()}")

print("=== LOADING AND CHECKING ACTUAL COLUMNS ===")

# Load the raw data first
df_raw = pd.read_csv('/kaggle/input/usa-real-estate/realtor-data.zip.csv')
print(f"Loaded: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

print(f"\nACTUAL COLUMN NAMES:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"{i:2}. '{col}'")

print(f"\nFIRST FEW ROWS:")
print(df_raw.head(2))

print("=== DATA TYPE CONVERSION (FIXED) ===")

df_clean = df_raw.copy()

print("Converting data types based on actual column names...")

# String/Categorical fields
df_clean['brokered_by'] = df_clean['brokered_by'].astype('string')  # Fixed name!
df_clean['status'] = df_clean['status'].astype('category')  
df_clean['street'] = df_clean['street'].astype('string')  # Street ID codes
df_clean['city'] = df_clean['city'].astype('string')
df_clean['state'] = df_clean['state'].astype('category')
df_clean['zip_code'] = df_clean['zip_code'].astype('string')

# Integer fields (with nullable support)
df_clean['bed'] = df_clean['bed'].astype('Int64')
df_clean['bath'] = df_clean['bath'].astype('Int64') 
df_clean['house_size'] = df_clean['house_size'].astype('Int64')

# Float fields  
df_clean['price'] = df_clean['price'].astype('float64')
df_clean['acre_lot'] = df_clean['acre_lot'].astype('float64')

# Handle dates properly
print("Converting dates...")
df_clean['prev_sold_date'] = pd.to_datetime(df_clean['prev_sold_date'], errors='coerce')

print(f"Data types converted!")
print(f"Original memory: {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"New memory: {df_clean.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\nNEW DATA TYPES:")
print(df_clean.dtypes)

print(f"\nSAMPLE OF CLEANED DATA:")
print(df_clean.head(2))

print("=== COMPLETE DATA QUALITY ANALYSIS - ALL 12 COLUMNS ===")

print("🔍 MISSING VALUES ANALYSIS (ALL COLUMNS):")
missing_summary = df_clean.isnull().sum()
missing_percent = (missing_summary / len(df_clean)) * 100
print("Column".ljust(15), "Missing Count".ljust(15), "Missing %")
print("-" * 45)
for col in df_clean.columns:
    print(f"{col:<15} {missing_summary[col]:<15,} {missing_percent[col]:<8.1f}%")

print(f"\n" + "="*60)
print("COLUMN-BY-COLUMN DETAILED ANALYSIS:")
print("="*60)

for col in df_clean.columns:
    print(f"\n🔍 {col.upper()}")
    print("-" * 30)
    print(f"Data type: {df_clean[col].dtype}")
    print(f"Missing: {df_clean[col].isnull().sum():,} ({missing_percent[col]:.1f}%)")
    print(f"Unique values: {df_clean[col].nunique():,}")
    
    if df_clean[col].dtype in ['float64', 'int64', 'Int64']:
        # Numeric analysis
        print(f"Min: {df_clean[col].min()}")
        print(f"Max: {df_clean[col].max()}")
        print(f"Mean: {df_clean[col].mean():.2f}")
        print(f"Median: {df_clean[col].median():.2f}")
        
        # Check for suspicious values
        if col == 'price':
            print(f"Zero prices: {(df_clean[col] == 0).sum():,}")
            print(f"Over $50M: {(df_clean[col] > 50_000_000).sum():,}")
        elif col == 'bed':
            print(f"Negative beds: {(df_clean[col] < 0).sum():,}")
            print(f"Over 20 beds: {(df_clean[col] > 20).sum():,}")
        elif col == 'bath':
            print(f"Negative baths: {(df_clean[col] < 0).sum():,}")
            print(f"Over 15 baths: {(df_clean[col] > 15).sum():,}")
        elif col == 'house_size':
            print(f"Under 100 sqft: {(df_clean[col] < 100).sum():,}")
            print(f"Over 50K sqft: {(df_clean[col] > 50000).sum():,}")
        elif col == 'acre_lot':
            print(f"Over 100 acres: {(df_clean[col] > 100).sum():,}")
            
    elif df_clean[col].dtype in ['category', 'string']:
        # Categorical analysis
        print("Top 5 values:")
        print(df_clean[col].value_counts().head())
        
    elif df_clean[col].dtype == 'datetime64[ns]':
        # Date analysis
        print(f"Date range: {df_clean[col].min()} to {df_clean[col].max()}")
        print("Sample dates:")
        print(df_clean[col].dropna().head(3).tolist())

df["brokered_by"] = df["brokered_by"].astype(str).replace("nan", np.nan)
df = df.dropna(subset=["brokered_by"])  

df["status"] = df["status"].astype("category")

df["price"] = df["price"].fillna(df["price"].median())
df.loc[df["price"] == 0, "price"] = df["price"].median()  
df.loc[df["price"] > 5e7, "price"] = 5e7  

df["bed"] = df["bed"].fillna(df["bed"].median()).astype("Int64")
df.loc[df["bed"] > 20, "bed"] = 20  

df["bath"] = df["bath"].fillna(df["bath"].median()).astype("Int64")
df.loc[df["bath"] > 15, "bath"] = 15  

df["acre_lot"] = df["acre_lot"].fillna(df["acre_lot"].median())
df.loc[df["acre_lot"] > 100, "acre_lot"] = 100  

df["street"] = df["street"].astype(str).replace("nan", "Unknown")

df["city"] = df["city"].astype(str).replace("nan", "Unknown")

df["state"] = df["state"].astype(str).replace("nan", "Unknown").astype("category")

df["zip_code"] = df["zip_code"].astype(str).replace("nan", "Unknown")

df["house_size"] = df["house_size"].fillna(df["house_size"].median()).astype("Int64")
df.loc[df["house_size"] < 100, "house_size"] = 100  
df.loc[df["house_size"] > 50000, "house_size"] = 50000 

df["prev_sold_date"] = pd.to_datetime(df["prev_sold_date"], errors="coerce")

print(df.head())
print(df.dtypes)
print(df.info())
print(df.describe(include='all'))
print(df["prev_sold_date"].head(20))
print(df['prev_sold_date'].isna().sum(), "missing values")
print(df['prev_sold_date'].notna().sum(), "valid values")
df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("After removing duplicates:", df.duplicated().sum())
print("New shape:", df.shape)

num_cols = ["price", "area"]

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n{col} outliers count: {outliers.shape[0]}")
    print(f"{col} normal range: {lower_bound} to {upper_bound}")

df = df[df['price'] > 0]   
df = df[df['price'] < 10000000]

removed_outliers = 171097 - len(df[(df['price'] > 1127500)])
print("Removed outliers:", removed_outliers)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.to_csv("real_estate_cleaned.csv", index=False)
df = df.drop_duplicates()
print("Remaining duplicate rows:", df.duplicated().sum())
df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = pd.read_csv("/kaggle/input/usa-real-estate/realtor-data.zip.csv")
df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(df.info())
df['street'] = df['street'].astype(str)
df['brokered_by'] = df['brokered_by'].astype(str)
df['zip_code'] = df['zip_code'].astype(str)
print(df.dtypes)
df.isnull().sum()
df.describe(include='all')
df['city'].value_counts().head(10)
df['status'].value_counts()
df[['price', 'bed', 'bath', 'acre_lot', 'house_size']].describe()
df.corr(numeric_only=True)
sns.set(style='whitegrid', palette='muted', font_scale=1.1)
df = pd.read_csv('/kaggle/input/usa-real-estate/realtor-data.zip.csv', parse_dates=['prev_sold_date'])





print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.describe(include='object'))

df['brokered_by'] = df['brokered_by'].astype('object')
df['street'] = df['street'].astype('object')
df['zip_code'] = df['zip_code'].astype('object')

df['bed'] = df['bed'].fillna(0).astype(int)
df['bath'] = df['bath'].fillna(0) 

df = pd.read_csv('/kaggle/input/usa-real-estate/realtor-data.zip.csv', parse_dates=['prev_sold_date'])

print("===== INFO =====")
print(df.info())
print("\n===== HEAD =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

df['brokered_by'] = df['brokered_by'].astype('object')
df['street'] = df['street'].astype('object')
df['zip_code'] = df['zip_code'].astype('object')
df['bed'] = df['bed'].fillna(0).astype(int)
df['bath'] = df['bath'].fillna(0)

df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
df = df[df['prev_sold_date'].notna()]
df = df[(df['prev_sold_date'].dt.year >= 1950) & (df['prev_sold_date'].dt.year <= 2025)]

numeric_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size']
print("\n===== NUMERIC STATS =====")
print(df[numeric_cols].describe())

print("\n===== CORRELATION =====")
print(df[numeric_cols].corr())

categorical_cols = ['city', 'state', 'status', 'brokered_by']
for col in categorical_cols:
    print(f"\nTop 10 categories for {col}:")
    print(df[col].value_counts().head(10))

df['sold_year'] = df['prev_sold_date'].dt.year
avg_price_year = df.groupby('sold_year')['price'].mean()
print("\n===== AVG PRICE PER YEAR =====")
print(avg_price_year)

print("\n===== FINAL CHECKS =====")
print("Total rows:", len(df))
print("Total missing values per column:\n", df.isnull().sum())
print("Price min/max:", df['price'].min(), df['price'].max())
print("House size min/max:", df['house_size'].min(), df['house_size'].max())
print("Bedrooms min/max:", df['bed'].min(), df['bed'].max())
print("Bathrooms min/max:", df['bath'].min(), df['bath'].max())
print("Acre lot min/max:", df['acre_lot'].min(), df['acre_lot'].max())

df = df[df['price'] > 1000]  
df = df[df['bed'].between(1, 10)]
df = df[df['bath'].between(1, 10)]
df = df[df['house_size'].between(100, 20000)]  
df = df[df['acre_lot'] < 10] 
df['brokered_by'] = df['brokered_by'].fillna('Unknown')
df['city'] = df['city'].fillna('Unknown')
df['street'] = df['street'].fillna('Unknown')
df = df[df['price'].notna()]
df = df[df['price'] > 1000]
df = df[df['bed'].between(1, 10)]
df = df[df['bath'].between(1, 10)]
df = df[df['house_size'].between(100, 20000)]
df = df[df['acre_lot'] < 10] 

df['brokered_by'] = df['brokered_by'].fillna('Unknown')
df['city'] = df['city'].fillna('Unknown')
df['street'] = df['street'].fillna('Unknown')

df.to_csv('/kaggle/working/real_estate_cleaned.csv', index=False)

df = pd.read_csv('/kaggle/input/usa-real-estate/realtor-data.zip.csv', parse_dates=['prev_sold_date'])

df['brokered_by'] = df['brokered_by'].fillna('Unknown')
df['city'] = df['city'].fillna('Unknown')
df['street'] = df['street'].fillna('Unknown')

df = df[df['price'].notna()]

df['brokered_by'] = df['brokered_by'].astype('object')
df['street'] = df['street'].astype('object')
df['zip_code'] = df['zip_code'].astype('object')
df['bed'] = df['bed'].fillna(0).astype(int)
df['bath'] = df['bath'].fillna(0)

df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
df = df[df['prev_sold_date'].notna()]
df = df[(df['prev_sold_date'].dt.year >= 1950) & (df['prev_sold_date'].dt.year <= 2025)]
df['sold_year'] = df['prev_sold_date'].dt.year

df = df[df['price'] > 1000]
df = df[df['bed'].between(1, 10)]
df = df[df['bath'].between(1, 10)]
df = df[df['house_size'].between(100, 20000)]
df = df[df['acre_lot'] < 10]  

print("Total rows:", len(df))
print("Missing values per column:\n", df.isnull().sum())
print("Price min/max:", df['price'].min(), df['price'].max())
print("House size min/max:", df['house_size'].min(), df['house_size'].max())
print("Bedrooms min/max:", df['bed'].min(), df['bed'].max())
print("Bathrooms min/max:", df['bath'].min(), df['bath'].max())
print("Acre lot min/max:", df['acre_lot'].min(), df['acre_lot'].max())

df.to_csv('/kaggle/working/real_estate_cleaned.csv', index=False)
print("\nCleaned dataset saved to /kaggle/working/real_estate_cleaned.csv")
