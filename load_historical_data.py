"""
Load Historical Cocoa Price Data from Multiple Sources.
Downloads 46 years of historical data from Yahoo Finance and imports CSV files.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Connect to Supabase
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))


def download_yahoo_finance_historical(years=46):
    """
    Download historical cocoa prices from Yahoo Finance.
    
    Args:
        years: Number of years of historical data (default: 46)
    
    Returns:
        DataFrame with historical prices
    """
    logger.info("=" * 80)
    logger.info(f"📥 DOWNLOADING {years} YEARS OF HISTORICAL DATA FROM YAHOO FINANCE")
    logger.info("=" * 80)
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        logger.info(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
        logger.info(f"End Date: {end_date.strftime('%Y-%m-%d')}")
        
        # Download data
        logger.info("Downloading data from Yahoo Finance (CC=F - Cocoa Futures)...")
        ticker = yf.Ticker("CC=F")
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.error("❌ No data downloaded from Yahoo Finance")
            return None
        
        # Reset index to get Date as a column
        df = df.reset_index()
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'price',
            'Volume': 'volume'
        })
        
        # Add metadata
        df['source'] = 'yahoo_finance_historical'
        df['symbol'] = 'CC=F'
        df['collected_at'] = df['date'].astype(str) + 'T12:00:00+00:00'
        
        # Select only needed columns
        df = df[['collected_at', 'source', 'symbol', 'price', 'open', 'high', 'low', 'volume', 'date']]
        
        # Convert date to string format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Remove rows with NaN prices
        df = df.dropna(subset=['price'])
        
        logger.info(f"✅ Downloaded {len(df)} data points")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error downloading Yahoo Finance data: {str(e)}")
        return None


def import_investing_com_csv(csv_path):
    """
    Import historical data from Investing.com CSV file.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with historical prices
    """
    logger.info("=" * 80)
    logger.info(f"📥 IMPORTING DATA FROM INVESTING.COM CSV")
    logger.info("=" * 80)
    
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"⚠️  CSV file not found: {csv_path}")
            return None
        
        logger.info(f"Reading CSV: {csv_path}")
        
        # Read CSV with proper format
        # Format: Date,"Price","Open","High","Low","Vol.","Change %"
        df = pd.read_csv(csv_path)
        
        logger.info(f"Columns found: {df.columns.tolist()}")
        
        # Clean column names (remove quotes and spaces)
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Parse date (format: MM/DD/YYYY)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        
        # Clean price columns (remove commas and convert to float)
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = df[col].str.replace(',', '').astype(float)
        
        # Clean volume (remove 'K', 'M' suffixes)
        if 'Vol.' in df.columns:
            df['Vol.'] = df['Vol.'].str.replace('K', '').str.replace('M', '').str.replace('-', '0')
            df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce') * 1000  # Assume K = thousands
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'Date': 'date',
            'Price': 'price',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Vol.': 'volume'
        })
        
        # Add metadata
        df['source'] = 'investing_com_historical'
        df['symbol'] = 'CCN6'
        df['collected_at'] = df['date'].astype(str) + 'T12:00:00+00:00'
        
        # Select only needed columns
        df = df[['collected_at', 'source', 'symbol', 'price', 'open', 'high', 'low', 'volume', 'date']]
        
        # Convert date to string format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Remove rows with NaN prices
        df = df.dropna(subset=['price'])
        
        # Sort by date (oldest first)
        df = df.sort_values('date')
        
        logger.info(f"✅ Imported {len(df)} data points")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error importing CSV: {str(e)}")
        logger.exception(e)
        return None


def save_to_supabase(df, batch_size=100):
    """
    Save historical data to Supabase in batches.
    
    Args:
        df: DataFrame with historical prices
        batch_size: Number of rows to insert per batch
    
    Returns:
        Number of rows inserted
    """
    logger.info("=" * 80)
    logger.info(f"💾 SAVING {len(df)} ROWS TO SUPABASE")
    logger.info("=" * 80)
    
    try:
        # Clean data before saving
        logger.info("🧹 Cleaning data...")
        
        # 1. Convert volume to integer (handle NaN and float values)
        df['volume'] = df['volume'].fillna(0).astype(float).astype(int)
        
        # 2. Replace NaN and Inf values with None
        df = df.replace([float('inf'), float('-inf')], None)
        df = df.where(pd.notnull(df), None)
        
        # 3. Fix timestamp format (remove timezone info from date string)
        # collected_at should be just the date + T12:00:00+00:00
        df['collected_at'] = df['date'] + 'T12:00:00+00:00'
        
        # 4. Ensure numeric columns are proper types
        for col in ['price', 'open', 'high', 'low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. Remove rows with missing critical data
        df = df.dropna(subset=['price', 'date'])
        
        logger.info(f"✅ Data cleaned. {len(df)} rows ready to insert.")
        
        # Convert DataFrame to list of dicts
        records = df.to_dict('records')
        
        # Insert in batches
        total_inserted = 0
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                client.table('cocoa_prices').insert(batch).execute()
                total_inserted += len(batch)
                logger.info(f"✅ Batch {batch_num}/{total_batches}: Inserted {len(batch)} rows ({total_inserted}/{len(records)})")
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {str(e)}")
                # Continue with next batch
        
        logger.info(f"✅ Total inserted: {total_inserted}/{len(records)} rows")
        return total_inserted
        
    except Exception as e:
        logger.error(f"❌ Error saving to Supabase: {str(e)}")
        return 0


def main():
    """
    Main function to load all historical data.
    """
    logger.info("=" * 80)
    logger.info("🚀 HISTORICAL DATA LOADER")
    logger.info("=" * 80)
    
    all_data = []
    
    # 1. Download Yahoo Finance data (46 years)
    yahoo_df = download_yahoo_finance_historical(years=46)
    if yahoo_df is not None:
        all_data.append(yahoo_df)
    
    # 2. Import Investing.com CSV files (multiple files)
    csv_files = [
        'data/12-27-1979_11-30_1999.csv',  # 1979-1999
        'data/11-08-1999_10-02-2019.csv',  # 1999-2019
        'data/10-01-2019_05-07_2026.csv'   # 2019-2026
    ]
    
    # Also check for any other CSV files in data folder
    import glob
    csv_files.extend(glob.glob('data/*.csv'))
    csv_files = list(set(csv_files))  # Remove duplicates
    
    for csv_path in csv_files:
        if os.path.exists(csv_path):
            logger.info(f"\n📄 Processing: {csv_path}")
            investing_df = import_investing_com_csv(csv_path)
            if investing_df is not None:
                all_data.append(investing_df)
    
    # 3. Combine all data
    if not all_data:
        logger.error("❌ No data to save!")
        return
    
    logger.info("=" * 80)
    logger.info("📊 COMBINING DATA FROM ALL SOURCES")
    logger.info("=" * 80)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (same date and source)
    combined_df = combined_df.drop_duplicates(subset=['date', 'source'], keep='first')
    
    # Sort by date
    combined_df = combined_df.sort_values('date')
    
    logger.info(f"Total unique data points: {len(combined_df)}")
    logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    logger.info(f"Sources: {combined_df['source'].unique().tolist()}")
    
    # 4. Save to Supabase
    inserted = save_to_supabase(combined_df, batch_size=100)
    
    # 5. Summary
    logger.info("=" * 80)
    logger.info("✅ HISTORICAL DATA LOADING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total data points: {len(combined_df)}")
    logger.info(f"Successfully inserted: {inserted}")
    logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    logger.info(f"Years of data: {(pd.to_datetime(combined_df['date'].max()) - pd.to_datetime(combined_df['date'].min())).days / 365:.1f}")
    logger.info("=" * 80)
    
    # Save to CSV for backup
    backup_path = 'data/historical_data_backup.csv'
    os.makedirs('data', exist_ok=True)
    combined_df.to_csv(backup_path, index=False)
    logger.info(f"💾 Backup saved to: {backup_path}")
    
    logger.info("\n🎯 READY FOR ML TRAINING!")
    logger.info("Next step: python train_model.py")


if __name__ == "__main__":
    main()
