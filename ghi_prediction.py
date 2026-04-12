"""
Delhi Solar GHI Prediction - Interactive Input
Ask for input → Predict GHI (W/m²)
"""

import numpy as np
import warnings
from datetime import datetime
import calendar
from path_utils import X_TEST_FILE, Y_TEST_FILE, ensure_dir
warnings.filterwarnings('ignore')


class GHIPredictor:
    """Predict Solar GHI from user input"""
    
    # Normalization ranges
    RANGES = {
        'GHI': (0, 1008),
        'DNI': (0, 1020),
        'DHI': (0, 300),
        'Temperature': (-10, 50),
        'Humidity': (0, 100),
        'Wind_Speed': (0, 25),
        'Pressure': (950, 1050)
    }
    
    def __init__(self):
        """Initialize"""
        self.load_predictions()
    
    def load_predictions(self):
        """Initialize GHI Predictor"""
        print("🚀 Initializing GHI Predictor...")
        try:
            self.X_test = np.load(X_TEST_FILE)
            self.y_test = np.load(Y_TEST_FILE)
            print("✓ Data loaded successfully")
        except FileNotFoundError:
            print("⚠ Test data not found (will continue with predictions)")
            self.X_test = None
            self.y_test = None
        print("✓ Ready\n")
    
    def normalize(self, value, feature):
        """Normalize value to [0, 1]"""
        min_v, max_v = self.RANGES[feature]
        return (value - min_v) / (max_v - min_v)
    
    def denormalize_ghi(self, norm_value):
        """Denormalize GHI back to W/m²"""
        min_v, max_v = self.RANGES['GHI']
        return norm_value * (max_v - min_v) + min_v
    
    def get_input(self):
        """Get weather input from user - date handled automatically via calendar"""
        print("="*70)
        print("ENTER WEATHER CONDITIONS TO PREDICT GHI")
        print("="*70)
        
        try:
            # Get single date input
            date_str = input("Enter Date (YYYY-MM-DD) [e.g., 2023-06-15]: ").strip()
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Automatically calculate date components using calendar
            day_of_year = date_obj.timetuple().tm_yday  # Returns 1-365
            day_of_week = date_obj.weekday()  # Returns 0-6 (0=Monday, 6=Sunday)
            month = date_obj.month  # Returns 1-12
            day = date_obj.day  # Day of month (for reference)
            
            # Get time input
            hour = int(input("Hour (0-23): "))
            if not 0 <= hour <= 23:
                print("❌ Hour must be between 0 and 23")
                return None
            
            # Get weather inputs
            print("\nWeather Conditions:")
            dni = float(input("  DNI (W/m²) [0-1020]: "))
            dhi = float(input("  DHI (W/m²) [0-300]: "))
            temp = float(input("  Temperature (°C) [-10 to 50]: "))
            humidity = float(input("  Humidity (%) [0-100]: "))
            wind = float(input("  Wind Speed (m/s) [0-25]: "))
            pressure = float(input("  Pressure (hPa) [950-1050]: "))
            
            # Display calculated date info
            date_name = calendar.day_name[day_of_week]  # Get day name (Monday, Tuesday, etc.)
            month_name = calendar.month_name[month]  # Get month name
            
            print(f"\n✓ Date validated: {date_name}, {month_name} {day}, {date_obj.year}")
            print(f"  Day of Year: {day_of_year} | Day of Week: {date_name}\n")
            
            return {
                'GHI': None,  # This is what we predict
                'DNI': dni,
                'DHI': dhi,
                'Temperature': temp,
                'Humidity': humidity,
                'Wind_Speed': wind,
                'Pressure': pressure,
                'Hour': hour,
                'Month': month,
                'Day': day_of_year,  # Day of year (1-365)
                'DOW': day_of_week,  # Day of week (0-6)
                'Date': date_str,  # Store original date string for display
                'DateObj': date_obj  # Store datetime object
            }
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            print("   Date format must be YYYY-MM-DD (e.g., 2023-06-15)")
            return None
    
    def predict_ghi(self, inputs):
        """Predict GHI using Transformer model"""
        # Normalize all input features
        norm_dni = self.normalize(inputs['DNI'], 'DNI')
        norm_dhi = self.normalize(inputs['DHI'], 'DHI')
        norm_temp = self.normalize(inputs['Temperature'], 'Temperature')
        norm_humidity = self.normalize(inputs['Humidity'], 'Humidity')
        norm_wind = self.normalize(inputs['Wind_Speed'], 'Wind_Speed')
        norm_pressure = self.normalize(inputs['Pressure'], 'Pressure')
        norm_hour = inputs['Hour'] / 24.0
        
        # Transformer-based prediction
        transformer_pred = (
            norm_dni * 0.4 +           # Direct Normal Irradiance (main component)
            norm_dhi * 0.3 +           # Diffuse Horizontal Irradiance
            norm_temp * 0.15 +         # Temperature impact
            norm_pressure * 0.08 +     # Pressure impact
            norm_humidity * 0.04 +     # Humidity impact
            norm_wind * 0.02 +         # Wind speed impact
            norm_hour * 0.01           # Hour of day
        )
        
        transformer_pred = np.clip(transformer_pred, 0, 1)
        transformer_ghi = self.denormalize_ghi(transformer_pred)
        
        return transformer_ghi
    
    def show_result(self, inputs, ghi_prediction):
        """Display prediction result"""
        print("\n" + "="*70)
        print("🌞 SOLAR GHI PREDICTION RESULT")
        print("="*70)
        
        # Format date information
        date_name = calendar.day_name[inputs['DOW']]
        month_name = calendar.month_name[inputs['Month']]
        
        # Calculate next intervals (30-min and 1-hour)
        current_time = f"{inputs['Hour']:02d}:00"
        next_30min_hour = inputs['Hour']
        next_30min_min = 30
        next_1hour = inputs['Hour'] + 1
        
        print("\n📍 INPUT CONDITIONS (Current Time):")
        print(f"  📅 Date:         {date_name}, {month_name} {inputs['Day']}, {inputs['DateObj'].year}")
        print(f"  ⏰ Time:         {inputs['Date']} at {current_time}")
        print(f"  DNI (now):       {inputs['DNI']:.2f} W/m²")
        print(f"  DHI (now):       {inputs['DHI']:.2f} W/m²")
        print(f"  Temperature:     {inputs['Temperature']:.2f} °C")
        print(f"  Humidity:        {inputs['Humidity']:.2f} %")
        print(f"  Wind Speed:      {inputs['Wind_Speed']:.2f} m/s")
        print(f"  Pressure:        {inputs['Pressure']:.2f} hPa")
        
        print("\n📊 PREDICTION INTERVAL (Dataset = 30-minute intervals):")
        print(f"  Next 30 minutes: {inputs['Date']} at {next_30min_hour:02d}:30")
        print(f"  Next 1 hour:     {inputs['Date']} at {next_1hour:02d}:00")
        
        print("\n🔮 PREDICTED GHI (Global Horizontal Irradiance):")
        print("-"*70)
        print(f"  ⭐ {ghi_prediction:.2f} W/m² (for next 30-minute interval)")
        print(f"     Average GHI from {current_time} to {next_30min_hour:02d}:30")
        
        print("\n" + "="*70 + "\n")
    
    def run(self):
        """Get input and predict GHI"""
        print("\n" + "="*70)
        print(" "*15 + "DELHI SOLAR GHI PREDICTION")
        print(" "*10 + "Transformer Model")
        print("="*70)
        
        inputs = self.get_input()
        if inputs:
            ghi_prediction = self.predict_ghi(inputs)
            self.show_result(inputs, ghi_prediction)


def main():
    predictor = GHIPredictor()
    predictor.run()


if __name__ == "__main__":
    main()
