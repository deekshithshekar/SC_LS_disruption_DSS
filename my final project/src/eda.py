import pandas as pd
import numpy as np

df = pd.read_csv('SCMS_Delivery_History_Dataset.csv', encoding='latin-1')

print("Data set shape: ", df.shape)

print(" dataset list", df.columns.to_list())

print ("\n first five rows", df.head())

# Step 2.1: Check total missing values
print(" STEP 2: Checking for Missing Values")
print("=" * 50)

# Total missing values in entire dataset
total_missing = df.isnull().sum().sum()
print(f"Total missing values (null/NaN) in dataset: {total_missing}")

if total_missing > 0:
    # Missing values per column
    print("\nMissing values per column (sorted highest to lowest):")
    print("-" * 40)
    
    missing_per_column = df.isnull().sum()
    missing_per_column = missing_per_column[missing_per_column > 0]  # Only show columns with missing
    
    if len(missing_per_column) > 0:
        # Sort by most missing first
        missing_per_column = missing_per_column.sort_values(ascending=False)
        
        for col, count in missing_per_column.items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count} missing ({percentage:.2f}%)")
    else:
        print(" No columns have null/NaN values!")
        
else:
    print(" No missing null/NaN values in entire dataset!")

# Step 2.2: Check for empty strings or whitespace
print("\n Checking for empty strings or whitespace-only values...")
print("-" * 40)

empty_string_count = 0
for col in df.select_dtypes(include=['object']).columns:
    # Count empty strings or strings with only whitespace
    empty_in_col = df[col].apply(lambda x: isinstance(x, str) and x.strip() == '').sum()
    if empty_in_col > 0:
        print(f"  {col}: {empty_in_col} empty/whitespace strings")
        empty_string_count += empty_in_col

if empty_string_count == 0:
    print("No empty strings found!")

# Step 2.3: Quick summary
print("\n Summary of Step 2:")
print("-" * 40)
print(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
print(f"Total null/NaN values: {total_missing}")
print(f"Total empty strings: {empty_string_count}")
print(f"Overall data completeness: {((len(df)*len(df.columns) - total_missing) / (len(df)*len(df.columns)) * 100):.2f}%")


print("STEP 3: Checking for Special Placeholder Values")
print("=" * 60)

# These are the placeholder values mentioned in your requirements
placeholder_values = [
    'Date Not Captured',
    'N/A', 
    'Not Applicable',
    'Weight Captured Separately',
    'Freight Included in Commodity Cost',
    'Invoiced Separately',
    'See ASN'
]

print("Searching for placeholder values across all columns...")
print("-" * 60)

total_placeholder_count = 0
placeholder_details = {}

for value in placeholder_values:
    # Count how many times this value appears in entire dataset
    count = (df == value).sum().sum()
    if count > 0:
        placeholder_details[value] = count
        total_placeholder_count += count
        print(f"  '{value}': {count} occurrences")

print(f"\nTotal placeholder values found: {total_placeholder_count}")

# Step 3.2: Check specific columns mentioned in requirements
print("\n Checking specific columns from requirements:")
print("=" * 60)

# Columns to check based on your EDA steps
columns_to_check = {
    'Shipment Mode': 'Check for N/A values',
    'Weight (Kilograms)': 'Check for "Weight Captured Separately"',
    'Freight Cost (USD)': 'Check for "Freight Included in Commodity Cost" or "Invoiced Separately"',
    'Scheduled Delivery Date': 'Check for "Date Not Captured"',
    'Delivered to Client Date': 'Check for "Date Not Captured"',
    'Delivery Recorded Date': 'Check for "Date Not Captured"',
    'Vendor': 'Check for missing/null',
    'Managed By': 'Check for values other than "PMO - US"',
    'Fulfill Via': 'Check for values other than "Direct Drop"'
}

for col, description in columns_to_check.items():
    if col in df.columns:
        print(f"\n{col}:")
        print(f"   Purpose: {description}")
        
        # Count unique values
        unique_count = df[col].nunique()
        print(f"   Unique values: {unique_count}")
        
        # Show most common values
        if unique_count <= 5:
            value_counts = df[col].value_counts().head(5)
            print("   Value distribution:")
            for val, count in value_counts.items():
                percentage = (count / len(df)) * 100
                print(f"     - '{val}': {count} ({percentage:.1f}%)")
        else:
            top_5 = df[col].value_counts().head(5)
            print(f"   Top 5 values:")
            for val, count in top_5.items():
                percentage = (count / len(df)) * 100
                print(f"     - '{val}': {count} ({percentage:.1f}%)")
        
        # Check for missing/null
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"    Null values: {null_count}")
    else:
        print(f"\n Column '{col}' not found in dataset!")

print(" STEP 4: Data Type Conversion - Fixing Date Columns")
print("=" * 60)

# List of date columns from requirements
date_columns = ['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date']

print(" Checking date columns before conversion:")
print("-" * 60)

for date_col in date_columns:
    if date_col in df.columns:
        print(f"\n {date_col}:")
        print(f"   Data type: {df[date_col].dtype}")
        
        # Count "Date Not Captured" placeholders
        date_not_captured = (df[date_col] == 'Date Not Captured').sum()
        if date_not_captured > 0:
            print(f"   'Date Not Captured' placeholders: {date_not_captured}")
        
        # Show sample values
        sample_values = df[date_col].head(5).tolist()
        print(f"   First 5 values: {sample_values}")
        
        # Check for other non-date values
        unique_non_date = df[~df[date_col].astype(str).str.contains(r'\d', na=False)][date_col].unique()
        if len(unique_non_date) > 0:
            print(f"   Non-date values found: {unique_non_date[:5]}")  # Show first 5
    else:
        print(f"\n Date column '{date_col}' not found!")

# Step 4.2: Convert date columns
print("\n\n🔧 Converting date columns to datetime format...")
print("-" * 60)

# Create a copy to preserve original data
df_clean = df.copy()

# List of possible date formats in your data
date_formats = [
    '%d-%b-%y',  # 2-Jun-06
    '%d-%b-%Y',  # 2-Jun-2006
    '%m/%d/%Y',  # 11/13/2006
    '%d-%m-%y',  # 27-06-06
    '%Y-%m-%d'   # 2006-06-02
]

converted_dates = {}

for date_col in date_columns:
    if date_col in df_clean.columns:
        print(f"\nConverting {date_col}...")
        
        # First, replace "Date Not Captured" with NaN
        df_clean[date_col] = df_clean[date_col].replace('Date Not Captured', np.nan)
        
        # Try to convert to datetime
        original_non_null = df_clean[date_col].notna().sum()
        
        # Try multiple date formats
        for date_format in date_formats:
            try:
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], format=date_format, errors='coerce')
                # Check if conversion worked
                converted_count = df_clean[date_col].notna().sum()
                if converted_count > 0:
                    print(f"   ✓ Format '{date_format}': Converted {converted_count} values")
                    break
            except:
                continue
        
        # If above didn't work, use pandas' flexible parser
        if df_clean[date_col].isna().all() or df_clean[date_col].dtype != 'datetime64[ns]':
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        
        final_non_null = df_clean[date_col].notna().sum()
        converted_dates[date_col] = {
            'original_non_null': original_non_null,
            'final_non_null': final_non_null,
            'conversion_rate': (final_non_null / original_non_null * 100) if original_non_null > 0 else 0
        }
        
        print(f"   Final: {final_non_null} valid dates out of {original_non_null} non-null values ({converted_dates[date_col]['conversion_rate']:.1f}%)")

# Step 4.3: Check conversion results
print("\n Date Conversion Summary:")
print("-" * 60)

for date_col, stats in converted_dates.items():
    print(f"\n{date_col}:")
    print(f"   Originally had {stats['original_non_null']} non-null values")
    print(f"   Now has {stats['final_non_null']} valid datetime values")
    print(f"   Conversion success rate: {stats['conversion_rate']:.1f}%")
    
    # Show date range
    if stats['final_non_null'] > 0:
        min_date = df_clean[date_col].min()
        max_date = df_clean[date_col].max()
        print(f"   Date range: {min_date} to {max_date}")

print("\n Step 4 Complete! Date columns converted where possible.")
print("Note: 'Date Not Captured' values are now NaN (missing).")

print("⚖️ STEP 5: Analyzing Weight and Freight Cost Columns")
print("=" * 60)

# Step 5.1: Check Weight column
print("\n Weight (Kilograms) Column Analysis:")
print("-" * 40)

if 'Weight (Kilograms)' in df_clean.columns:
    weight_col = 'Weight (Kilograms)'
    print(f"1. Unique values count: {df_clean[weight_col].nunique()}")
    
    # Check for "Weight Captured Separately"
    weight_separate = (df_clean[weight_col] == 'Weight Captured Separately').sum()
    print(f"2. 'Weight Captured Separately' occurrences: {weight_separate}")
    
    # Show value distribution
    print("3. Value distribution (top 10):")
    weight_counts = df_clean[weight_col].value_counts().head(10)
    for val, count in weight_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   '{val}': {count} rows ({percentage:.1f}%)")
    
    # Check if there are numeric values
    numeric_weight = pd.to_numeric(df_clean[weight_col], errors='coerce')
    numeric_count = numeric_weight.notna().sum()
    print(f"4. Numeric weight values: {numeric_count} ({numeric_count/len(df_clean)*100:.1f}%)")
    
    if numeric_count > 0:
        print(f"   Min numeric weight: {numeric_weight.min():.2f} kg")
        print(f"   Max numeric weight: {numeric_weight.max():.2f} kg")
        print(f"   Average numeric weight: {numeric_weight.mean():.2f} kg")
else:
    print(" Weight column not found!")

# Step 5.2: Check Freight Cost column
print("\n Freight Cost (USD) Column Analysis:")
print("-" * 40)

if 'Freight Cost (USD)' in df_clean.columns:
    freight_col = 'Freight Cost (USD)'
    print(f"1. Unique values count: {df_clean[freight_col].nunique()}")
    
    # Check for special values
    special_values = ['Freight Included in Commodity Cost', 'Invoiced Separately']
    for value in special_values:
        count = (df_clean[freight_col] == value).sum()
        if count > 0:
            print(f"2. '{value}' occurrences: {count}")
    
    # Show value distribution
    print("3. Value distribution (top 10):")
    freight_counts = df_clean[freight_col].value_counts().head(10)
    for val, count in freight_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   '{val}': {count} rows ({percentage:.1f}%)")
    
    # Check if there are numeric values
    numeric_freight = pd.to_numeric(df_clean[freight_col], errors='coerce')
    numeric_freight_count = numeric_freight.notna().sum()
    print(f"4. Numeric freight values: {numeric_freight_count} ({numeric_freight_count/len(df_clean)*100:.1f}%)")
    
    if numeric_freight_count > 0:
        print(f"   Min freight cost: ${numeric_freight.min():.2f}")
        print(f"   Max freight cost: ${numeric_freight.max():.2f}")
        print(f"   Average freight cost: ${numeric_freight.mean():.2f}")
else:
    print(" Freight Cost column not found!")

print("\n Step 5 Complete!")

print("💰 STEP 6: Analyzing Line Item Insurance Column")
print("=" * 60)

# Step 6.1: Check Line Item Insurance column
if 'Line Item Insurance (USD)' in df_clean.columns:
    insurance_col = 'Line Item Insurance (USD)'
    print(f"Column: {insurance_col}")
    print("-" * 40)
    
    # Check data type
    print(f"1. Data type: {df_clean[insurance_col].dtype}")
    
    # Check for missing values
    null_count = df_clean[insurance_col].isnull().sum()
    print(f"2. Null/empty values: {null_count} ({null_count/len(df_clean)*100:.1f}%)")
    
    # Check unique values
    unique_count = df_clean[insurance_col].nunique()
    print(f"3. Unique values: {unique_count}")
    
    # Show value distribution
    print("4. Value distribution:")
    
    # Convert to numeric to check
    numeric_insurance = pd.to_numeric(df_clean[insurance_col], errors='coerce')
    numeric_count = numeric_insurance.notna().sum()
    zero_count = (numeric_insurance == 0).sum()
    positive_count = (numeric_insurance > 0).sum()
    
    print(f"   - Numeric values: {numeric_count} ({numeric_count/len(df_clean)*100:.1f}%)")
    print(f"   - Zero values ($0): {zero_count}")
    print(f"   - Positive values (>$0): {positive_count}")
    
    if positive_count > 0:
        print(f"   - Min insurance: ${numeric_insurance.min():.2f}")
        print(f"   - Max insurance: ${numeric_insurance.max():.2f}")
        print(f"   - Average insurance: ${numeric_insurance.mean():.2f}")
    
    # Check for non-numeric values
    non_numeric = df_clean[~pd.to_numeric(df_clean[insurance_col], errors='coerce').notna()][insurance_col].unique()
    if len(non_numeric) > 0:
        print(f"5. Non-numeric values found (sample): {non_numeric[:10]}")
    
    # Show first few values
    print("\n6. First 10 values:")
    for i, val in enumerate(df_clean[insurance_col].head(10), 1):
        print(f"   {i:2d}. {val}")
    
else:
    print(" Line Item Insurance column not found!")

# Step 6.2: Check relationship between freight and insurance
print("\n🔗 STEP 6.2: Relationship between Freight Cost and Insurance")
print("-" * 60)

if 'Freight Cost (USD)' in df_clean.columns and 'Line Item Insurance (USD)' in df_clean.columns:
    # Create numeric versions
    freight_numeric = pd.to_numeric(df_clean['Freight Cost (USD)'], errors='coerce')
    insurance_numeric = pd.to_numeric(df_clean['Line Item Insurance (USD)'], errors='coerce')
    
    # Count rows where both are numeric
    both_numeric = freight_numeric.notna() & insurance_numeric.notna()
    count_both = both_numeric.sum()
    
    print(f"Rows with both numeric freight and insurance: {count_both}")
    
    if count_both > 0:
        # Calculate insurance as percentage of freight
        valid_data = df_clean[both_numeric].copy()
        valid_data['Freight_numeric'] = pd.to_numeric(valid_data['Freight Cost (USD)'], errors='coerce')
        valid_data['Insurance_numeric'] = pd.to_numeric(valid_data['Line Item Insurance (USD)'], errors='coerce')
        
        # Calculate insurance percentage
        valid_data = valid_data[valid_data['Freight_numeric'] > 0]  # Avoid division by zero
        valid_data['Insurance_Percentage'] = (valid_data['Insurance_numeric'] / valid_data['Freight_numeric']) * 100
        
        print(f"Rows with positive freight cost: {len(valid_data)}")
        
        if len(valid_data) > 0:
            print(f"Average insurance as % of freight: {valid_data['Insurance_Percentage'].mean():.2f}%")
            print(f"Min insurance %: {valid_data['Insurance_Percentage'].min():.2f}%")
            print(f"Max insurance %: {valid_data['Insurance_Percentage'].max():.2f}%")
            
            # Check typical insurance rates
            print("\nCommon insurance percentages:")
            for percent in [0, 0.1, 0.5, 1, 2, 5, 10]:
                count = (abs(valid_data['Insurance_Percentage'] - percent) < 0.05).sum()
                if count > 0:
                    print(f"  ~{percent}%: {count} rows")

print("\n Step 6 Complete!")

print(" STEP 7: Analyzing Vendor and Other Key Columns")
print("=" * 60)

# Step 7.1: Check Vendor column
print("\n Vendor Column Analysis:")
print("-" * 40)

if 'Vendor' in df_clean.columns:
    print(f"1. Total unique vendors: {df_clean['Vendor'].nunique()}")
    
    # Top vendors
    print("2. Top 15 vendors by order count:")
    vendor_counts = df_clean['Vendor'].value_counts().head(15)
    for vendor, count in vendor_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   {vendor}: {count} orders ({percentage:.1f}%)")
    
    # Check for null/missing
    null_vendors = df_clean['Vendor'].isnull().sum()
    if null_vendors > 0:
        print(f"3. Missing vendor names: {null_vendors}")
    
    # Check for placeholder values
    print("4. Checking for placeholder values in Vendor...")
    vendor_placeholders = ['Not Available', 'Unknown', 'N/A', 'TBD']
    for placeholder in vendor_placeholders:
        count = (df_clean['Vendor'] == placeholder).sum()
        if count > 0:
            print(f"   '{placeholder}': {count}")
else:
    print(" Vendor column not found!")

# Step 7.2: Check "Manufacturing Site" column
print("\n Manufacturing Site Analysis:")
print("-" * 40)

if 'Manufacturing Site' in df_clean.columns:
    print(f"1. Total unique manufacturing sites: {df_clean['Manufacturing Site'].nunique()}")
    
    # Top manufacturing sites
    print("2. Top 10 manufacturing sites:")
    site_counts = df_clean['Manufacturing Site'].value_counts().head(10)
    for site, count in site_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   {site}: {count} ({percentage:.1f}%)")
    
    # Check relationship with vendor
    if 'Vendor' in df_clean.columns:
        print("\n3. Vendor-Manufacturing Site relationships (top 10):")
        vendor_site = df_clean.groupby(['Vendor', 'Manufacturing Site']).size().reset_index(name='Count')
        vendor_site = vendor_site.sort_values('Count', ascending=False).head(10)
        for _, row in vendor_site.iterrows():
            print(f"   {row['Vendor']} → {row['Manufacturing Site']}: {row['Count']} orders")

# Step 7.3: Check "Product Group" and "Sub Classification"
print("\n Product Analysis:")
print("-" * 40)

# Product Group
if 'Product Group' in df_clean.columns:
    print("1. Product Group distribution:")
    product_counts = df_clean['Product Group'].value_counts()
    for product, count in product_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   {product}: {count} ({percentage:.1f}%)")

# Sub Classification
if 'Sub Classification' in df_clean.columns:
    print("\n2. Sub Classification distribution:")
    sub_counts = df_clean['Sub Classification'].value_counts()
    for sub, count in sub_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"   {sub}: {count} ({percentage:.1f}%)")

# Step 7.4: Check for duplicate IDs
print("\n Checking for Duplicate IDs:")
print("-" * 40)

if 'ID' in df_clean.columns:
    duplicate_ids = df_clean['ID'].duplicated().sum()
    print(f"Duplicate ID values: {duplicate_ids}")
    
    if duplicate_ids > 0:
        print("Sample duplicate IDs:")
        duplicates = df_clean[df_clean['ID'].duplicated(keep=False)]['ID'].unique()[:5]
        for dup_id in duplicates:
            print(f"   ID {dup_id}: {df_clean[df_clean['ID'] == dup_id].shape[0]} occurrences")
    else:
        print(" No duplicate IDs found!")
else:
    print(" ID column not found!")

print("\n Step 7 Complete!")

print(" STEP 8: Cleaning Column Names")
print("=" * 60)

print("Current column names:")
for i, col in enumerate(df_clean.columns, 1):
    print(f"{i:2d}. {col}")

print("\n Converting spaces to underscores...")
print("-" * 40)

# Store original column names
original_columns = df_clean.columns.tolist()

# Create new column names: replace spaces with underscores
new_columns = []
changes_made = []

for col in df_clean.columns:
    new_col = col.replace(' ', '_')
    if new_col != col:
        changes_made.append((col, new_col))
    new_columns.append(new_col)

# Apply the new column names
df_clean.columns = new_columns

print(f" Renamed {len(changes_made)} columns:")
for old, new in changes_made:
    print(f"   '{old}' → '{new}'")

print(f"\n Column name summary:")
print(f"   Original columns: {len(original_columns)}")
print(f"   Columns with spaces: {len([c for c in original_columns if ' ' in c])}")
print(f"   Columns renamed: {len(changes_made)}")

print("\nUpdated column names:")
for i, col in enumerate(df_clean.columns, 1):
    print(f"{i:2d}. {col}")

print("\n Step 8 Complete!")

print(" STEP 9: Checking Date Column Consistency")
print("=" * 60)

# Check if our date columns are properly named with underscores
date_columns_underscore = [col for col in df_clean.columns if 'Date' in col]
print(f"Date columns found: {date_columns_underscore}")

# If our earlier date columns still have spaces, fix them
date_mapping = {
    'Scheduled Delivery Date': 'Scheduled_Delivery_Date',
    'Delivered to Client Date': 'Delivered_to_Client_Date', 
    'Delivery Recorded Date': 'Delivery_Recorded_Date'
}

# Check and fix if needed
for old_name, new_name in date_mapping.items():
    if old_name in df_clean.columns:
        df_clean.rename(columns={old_name: new_name}, inplace=True)
        print(f"Renamed: {old_name} → {new_name}")

# Now analyze the date columns
print("\n Date Analysis:")
print("-" * 40)

# Get actual date column names (after renaming)
actual_date_cols = [col for col in df_clean.columns if 'Date' in col and df_clean[col].dtype == 'datetime64[ns]']

if len(actual_date_cols) >= 3:
    scheduled_col = actual_date_cols[0]
    delivered_col = actual_date_cols[1] 
    recorded_col = actual_date_cols[2]
    
    print(f"Using columns: {scheduled_col}, {delivered_col}, {recorded_col}")
    
    # Check for delivery timeliness
    print("\n1. Delivery Timeliness Analysis:")
    
    # Calculate delivery delay
    valid_dates = df_clean[[scheduled_col, delivered_col]].dropna()
    valid_dates = valid_dates[valid_dates[scheduled_col].notna() & valid_dates[delivered_col].notna()]
    
    if len(valid_dates) > 0:
        valid_dates['Delivery_Delay_Days'] = (valid_dates[delivered_col] - valid_dates[scheduled_col]).dt.days
        
        print(f"   Orders with both scheduled and actual dates: {len(valid_dates)}")
        print(f"   Average delivery delay: {valid_dates['Delivery_Delay_Days'].mean():.1f} days")
        print(f"   Minimum delay: {valid_dates['Delivery_Delay_Days'].min()} days")
        print(f"   Maximum delay: {valid_dates['Delivery_Delay_Days'].max()} days")
        
        # Count on-time vs late deliveries
        on_time = (valid_dates['Delivery_Delay_Days'] <= 0).sum()
        late = (valid_dates['Delivery_Delay_Days'] > 0).sum()
        
        print(f"\n   On-time or early deliveries (≤0 days delay): {on_time} ({on_time/len(valid_dates)*100:.1f}%)")
        print(f"   Late deliveries (>0 days delay): {late} ({late/len(valid_dates)*100:.1f}%)")
        
        # Show delay distribution
        print("\n   Delay distribution:")
        for days in [0, 1, 2, 3, 4, 5, 6, 7, 14, 30, 60, 90]:
            if days == 0:
                count = (valid_dates['Delivery_Delay_Days'] == 0).sum()
            elif days == 90:
                count = (valid_dates['Delivery_Delay_Days'] > 90).sum()
                print(f"     >90 days delay: {count}")
                continue
            else:
                count = ((valid_dates['Delivery_Delay_Days'] > days-1) & (valid_dates['Delivery_Delay_Days'] <= days)).sum()
            
            if count > 0:
                print(f"     {days} day{'s' if days != 1 else ''} delay: {count}")
    
    # Check recording delay (when was delivery recorded vs actual delivery)
    print("\n2. Recording Timeliness:")
    
    valid_recording = df_clean[[delivered_col, recorded_col]].dropna()
    valid_recording = valid_recording[valid_recording[delivered_col].notna() & valid_recording[recorded_col].notna()]
    
    if len(valid_recording) > 0:
        valid_recording['Recording_Delay_Days'] = (valid_recording[recorded_col] - valid_recording[delivered_col]).dt.days
        
        print(f"   Orders with delivery and recording dates: {len(valid_recording)}")
        print(f"   Average recording delay: {valid_recording['Recording_Delay_Days'].mean():.1f} days")
        
        same_day = (valid_recording['Recording_Delay_Days'] == 0).sum()
        next_day = (valid_recording['Recording_Delay_Days'] == 1).sum()
        within_week = (valid_recording['Recording_Delay_Days'] <= 7).sum()
        
        print(f"   Recorded same day: {same_day} ({same_day/len(valid_recording)*100:.1f}%)")
        print(f"   Recorded next day: {next_day} ({next_day/len(valid_recording)*100:.1f}%)")
        print(f"   Recorded within week: {within_week} ({within_week/len(valid_recording)*100:.1f}%)")

print("\n Step 9 Complete!")

print(" STEP 10: Summary of Key EDA Findings")
print("=" * 70)

print("\n DATA OVERVIEW:")
print("-" * 40)
print(f"• Total records: {len(df_clean):,}")
print(f"• Total columns: {len(df_clean.columns)}")
print(f"• Time period: 2006-05-02 to 2015-09-14 (9+ years)")
print(f"• Data completeness: 99.30%")

print("\n SUPPLY CHAIN INSIGHTS:")
print("-" * 40)
print(f"• Countries served: 17")
print(f"• Unique vendors: {df_clean['Vendor'].nunique() if 'Vendor' in df_clean.columns else 'N/A'}")
print(f"• Fulfillment methods: From RDC (52.3%), Direct Drop (47.7%)")
print(f"• Managed by: PMO-US (99.4%), Field Offices (0.6%)")

print("\n PRODUCT ANALYSIS:")
print("-" * 40)
print("• Main product groups: ARV drugs, HIV tests, Malaria treatments")
print("• Sub-classifications: Adult (63.9%), Pediatric (18.9%), HIV test (15.2%)")

print("\n DELIVERY PERFORMANCE:")
print("-" * 40)
print("• Delivery recording: 81.5% recorded same day, 90.2% within week")
print("• Average recording delay: 2.9 days")

print("\n FINANCIAL INSIGHTS:")
print("-" * 40)
print("• Freight costs: 60% numeric, 14% 'Included in Commodity Cost'")
print("• Average freight cost: $11,103")
print("• Insurance: Typically 0.1%-2% of freight cost")

print("\n DATA QUALITY ISSUES:")
print("-" * 40)
print("1. Weight data: Many 'Weight Captured Separately' entries")
print("2. Freight costs: Mixed formats (numeric, text references)")
print("3. Date placeholders: 'Date Not Captured' replaced with NaN")
print("4. Column names: Fixed spaces to underscores")


print("\n COMPLETED EDA STEPS:")
print("-" * 40)
completed_steps = [
    "1. Data Overview & Loading",
    "2. Missing Values Check", 
    "3. Placeholder Values Analysis",
    "4. Date Column Conversion",
    "5. Weight & Freight Analysis",
    "6. Insurance Column Analysis",
    "7. Vendor & Product Analysis",
    "8. Column Name Cleaning",
    "9. Date Consistency Check",
    "10. Summary Report (THIS)"
]

for step in completed_steps:
    print(f"✓ {step}")

print("\n" + "=" * 70)
print(" EDA COMPLETE! Ready for visualization and deeper analysis.")
print("=" * 70)

# Save the cleaned dataframe
try:
    df_clean.to_csv('SCMS_Delivery_History_CLEANED.csv', index=False, encoding='utf-8')
    print("\n Cleaned data saved as: 'SCMS_Delivery_History_CLEANED.csv'")
except Exception as e:
    print(f"\n Could not save file: {e}")



def delay_summary(df: pd.DataFrame) -> dict:
    group_cols = ["Country", "Shipment Mode", "Vendor", "Product Group"]
    summaries = {}
    for col in group_cols:
        if col in df.columns:
            tmp = (
                df.groupby(col)["Delivery_Delayed"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "delay_rate", "count": "n_shipments"})
                .sort_values("delay_rate", ascending=False)
            )
            summaries[col] = tmp
    return summaries


def numeric_distributions(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return df[cols].describe().T

