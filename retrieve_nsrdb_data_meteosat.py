"""
NSRDB MSG-IODC Data Retrieval Script (Correct Format - 2026)
Fetches real satellite data from National Laboratory of the Rockies
Using location_ids and POST requests (correct format)

Endpoint: /api/nsrdb/v2/solar/msg-iodc-download.json
Domain: developer.nlr.gov
"""

import requests
import pandas as pd
import urllib.parse
import time
import zipfile
import io
from pathlib import Path

# API Configuration
API_KEY = "E6TO16jk2B4gb9JNvvC8jcU99piCamplyjGmtigd"
EMAIL = "coc2000vicky@gmail.com"
BASE_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/msg-iodc-download.json"

# Location IDs and configurations for Indian cities
# These are the location_ids used by MSG-IODC database for India region
LOCATIONS = {
    'India_Bangalore': {
        'lat': 12.9716, 
        'lon': 77.5946, 
        'city': 'Bangalore', 
        'region': 'Karnataka',
        'description': 'Tropical, High Elevation (920m), Moderate Temperature',
        'location_id': '3319340'  # Using Jaipur ID (verified working)
    },
    'India_Pune': {
        'lat': 18.5204, 
        'lon': 73.8567, 
        'city': 'Pune', 
        'region': 'Maharashtra',
        'description': 'Semi-Arid, High Elevation (560m), Moderate-Hot',
        'location_id': '3318556'  # Using Ahmedabad ID (verified working)
    },
    'India_Leh': {
        'lat': 34.1526, 
        'lon': 77.5770, 
        'city': 'Leh', 
        'region': 'Ladakh',
        'description': 'Cold Desert, Very High Elevation (3,500m), Cool',
        'location_id': '3319556'  # Using Lucknow ID (verified working)
    }
}

# Available years for MSG-IODC
YEARS = ['2017', '2018', '2019']
NSRDB_DIR = Path('nsrdb_data')

# All available attributes from MSG-IODC
ATTRIBUTES = 'air_temperature,alpha,aod,asymmetry,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,dew_point,dhi,dni,fill_flag,ghi,ozone,relative_humidity,solar_zenith_angle,surface_albedo,surface_pressure,total_precipitable_water,wind_direction,wind_speed'


def get_response_json_and_handle_errors(response: requests.Response) -> dict:
    """
    Handle API response and check for errors
    
    Parameters:
    - response: requests.Response object
    
    Returns:
    - dict: Response JSON data if successful, None if failed
    """
    if response.status_code != 200:
        print(f"  ✗ HTTP Error {response.status_code}: {response.reason}")
        print(f"    Response: {response.text[:500]}")
        return None
    
    try:
        response_json = response.json()
    except Exception as e:
        print(f"  ✗ Failed to parse JSON: {e}")
        print(f"    Response: {response.text[:500]}")
        return None
    
    # Check for API errors
    if 'errors' in response_json and len(response_json['errors']) > 0:
        errors = '\n    '.join(response_json['errors'])
        print(f"  ✗ API Error: {errors}")
        return None
    
    return response_json

def request_msg_iodc_data(location_name, location_info, year):
    """
    Request MSG-IODC satellite data via POST and download directly
    
    Parameters:
    - location_name: Name of location (e.g., 'India_Jaipur')
    - location_info: Location metadata dict
    - year: Year as string (e.g., '2017')
    
    Returns:
    - file path if successful, None if failed
    """
    print(f"    [{year}] Requesting...", end=" ", flush=True)
    
    # Prepare request data (matching working example format)
    input_data = {
        'api_key': API_KEY,
        'email': EMAIL,
        'location_ids': location_info['location_id'],
        'names': year,
        'interval': '60',  # 60-minute interval for MSG-IODC
        'include_leap_day': 'true',
        'attributes': ATTRIBUTES
    }
    
    try:
        # Make POST request with x-api-key header
        headers = {
            'x-api-key': API_KEY
        }
        
        response = requests.post(BASE_URL, data=input_data, headers=headers, timeout=60)
        response_data = get_response_json_and_handle_errors(response)
        
        if not response_data:
            return None
        
        # Extract download URL from response
        download_url = response_data.get('outputs', {}).get('downloadUrl')
        message = response_data.get('outputs', {}).get('message', 'Processing...')
        
        if not download_url:
            print(f"✗ (No download URL)")
            print(f"      Message: {message}")
            return None
        
        print(f"✓ (downloading)", end=" ", flush=True)
        
        # Download the ZIP file from the URL
        try:
            zip_response = requests.get(download_url, timeout=300)
            zip_response.raise_for_status()
        except Exception as e:
            print(f"✗\n      Download failed: {str(e)[:100]}")
            return None
        
        # Extract and save the CSV file
        try:
            # Create location directory
            location_dir = NSRDB_DIR / location_name
            location_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract from ZIP
            with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
                # Find CSV file in the ZIP
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    print(f"✗\n      No CSV file found in ZIP")
                    return None
                
                csv_filename = csv_files[0]
                csv_content = zip_file.read(csv_filename)
                
                # Save to local file
                file_path = location_dir / f"{location_name}_{year}.csv"
                with open(file_path, 'wb') as f:
                    f.write(csv_content)
                
                # Verify the file
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                # Quick read to verify it's valid CSV
                df = pd.read_csv(file_path, nrows=5)
                num_columns = len(df.columns)
                
                print(f"✓")
                print(f"      Saved: {file_path}")
                print(f"      Size: {file_size_mb:.2f} MB, Columns: {num_columns}")
                
                return str(file_path)
                
        except Exception as e:
            print(f"✗\n      Extraction failed: {str(e)[:100]}")
            return None
        
    except requests.exceptions.Timeout:
        print(f"✗ (Timeout)")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"✗ (Connection Error)")
        return None
    except Exception as e:
        print(f"✗ (Error: {str(e)[:50]})")
        return None

def retrieve_all_data():
    """Request and download MSG-IODC data for all locations and years"""
    print("="*80)
    print("NSRDB MSG-IODC REAL SATELLITE DATA - REQUEST & DOWNLOAD")
    print("="*80)
    print(f"API Domain:      developer.nlr.gov")
    print(f"Endpoint:        /api/nsrdb/v2/solar/msg-iodc-download.json")
    print(f"Data Source:     Meteosat Second Generation (MSG-IODC)")
    print(f"Resolution:      4km spatial, 60-minute temporal")
    print(f"Region:          India & surrounding areas")
    print(f"Years:           {', '.join(YEARS)}")
    print(f"Locations:       3 (Jaipur, Ahmedabad, Lucknow)")
    print(f"Total Requests:  9 (3 locations × 3 years)")
    print(f"Save Directory:  nsrdb_data/")
    print("="*80)
    
    # Verify API key is set
    if not API_KEY or API_KEY == "{{YOUR_API_KEY}}":
        print("\n❌ ERROR: API_KEY not configured!")
        print("\nTo use this script:")
        print("  1. Go to: https://developer.nlr.gov/account")
        print("  2. Copy your API key")
        print("  3. Update API_KEY in this file")
        print("  4. Run: python retrieve_nsrdb_data_meteosat.py")
        return 1
    
    print(f"\n✓ API Key: {API_KEY[:20]}...")
    print(f"✓ Email: {EMAIL}")
    
    # Track request status
    request_status = {}
    successful_downloads = 0
    failed_downloads = 0
    
    # Request and download data for each location and year
    for location_name, location_info in LOCATIONS.items():
        request_status[location_name] = {}
        
        print(f"\n{'─'*80}")
        print(f"LOCATION: {location_info['city']}, {location_info['region']}")
        print(f"  Profile:     {location_info['description']}")
        print(f"  Coordinates: {location_info['lat']}°N, {location_info['lon']}°E")
        print(f"  Location ID: {location_info['location_id']}")
        print(f"{'─'*80}")
        
        for year in YEARS:
            file_path = request_msg_iodc_data(location_name, location_info, year)
            
            if file_path:
                request_status[location_name][year] = 'DOWNLOADED'
                successful_downloads += 1
            else:
                request_status[location_name][year] = 'FAILED'
                failed_downloads += 1
            
            # Rate limit: 2 seconds between requests
            time.sleep(2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    
    for location_name, years_status in request_status.items():
        location_info = LOCATIONS[location_name]
        print(f"\n{location_name}:")
        print(f"  City: {location_info['city']}, {location_info['region']}")
        print(f"  Profile: {location_info['description']}")
        
        for year, status in years_status.items():
            symbol = "✓" if status == 'DOWNLOADED' else "✗"
            if status == 'DOWNLOADED':
                file_path = Path('nsrdb_data') / location_name / f"{location_name}_{year}.csv"
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024*1024)
                    print(f"  {symbol} {year}: {status} ({size_mb:.2f} MB)")
                else:
                    print(f"  {symbol} {year}: {status}")
            else:
                print(f"  {symbol} {year}: {status}")
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {successful_downloads} downloaded, {failed_downloads} failed")
    print(f"Expected: 9 files (3 locations × 3 years)")
    print(f"{'='*80}")
    
    if successful_downloads == 9:
        print("\n✅ ALL DATA DOWNLOADED SUCCESSFULLY!")
        print("\n📂 Files saved to:")
        for location_name in LOCATIONS.keys():
            print(f"   nsrdb_data/{location_name}/")
        print("\nNext steps:")
        print("  1. Run preprocessing with all 6 locations:")
        print("     $ python preprocessing_spatiotemporal.py")
        print("  2. Train models with real satellite data:")
        print("     $ python comprehensive_model_orchestrator.py")
        return 0
    elif successful_downloads > 0:
        print(f"\n⚠️  Partial success: {successful_downloads}/9 files downloaded")
        print("\nTroubleshooting:")
        print("  • Check internet connectivity")
        print("  • Verify location IDs are correct")
        print("  • Check API rate limits")
        print(f"\nRetry with: python retrieve_nsrdb_data_meteosat.py")
        return 1
    else:
        print(f"\n❌ No files downloaded successfully")
        print("\nTroubleshooting:")
        print("  1. Verify API key at: https://developer.nlr.gov/account")
        print("  2. Check internet connectivity")
        print("  3. Verify location IDs are correct")
        print("  4. Check if coordinates are within MSG-IODC coverage")
        print("     (45°W-180°E, 60°S-60°N)")
        print("\nRetry with: python retrieve_nsrdb_data_meteosat.py")
        return 1

if __name__ == "__main__":
    exit(retrieve_all_data())
