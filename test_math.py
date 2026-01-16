# test_math.py
import numpy as np
from src.core.cga_math import CGAEngine, R_EARTH

def test_cga_basics():
    engine = CGAEngine()
    print("--- Testing CGA Engine ---")
    
    # 1. Test Mapping
    lat, lon, alt = 10.762, 106.660, 0.0 # Ho Chi Minh City
    p_hcm = engine.latlon_to_cga(lat, lon, alt)
    print(f"Point HCM created: {p_hcm}")
    
    # 2. Test Distance (HCM to Hanoi)
    lat_hn, lon_hn = 21.028, 105.854
    p_hn = engine.latlon_to_cga(lat_hn, lon_hn, 0.0)
    
    d_sq = engine.get_distance_sq(p_hcm, p_hn)
    d_km = np.sqrt(d_sq)
    print(f"Distance HCM - HN (Chord length): {d_km:.2f} km")
    # Khoảng cách dây cung (Chord) sẽ nhỏ hơn khoảng cách mặt cầu (Geodesic) một chút
    # HCM-HN đường chim bay khoảng 1137km. Chord length nên xấp xỉ.
    
    # 3. Test Satellite Position (Starlink Height ~550km)
    p_sat = engine.latlon_to_cga(lat, lon, 550.0) # Vệ tinh ngay trên đầu HCM
    d_sat_sq = engine.get_distance_sq(p_hcm, p_sat)
    print(f"Distance to Sat overhead: {np.sqrt(d_sat_sq):.2f} km (Expected ~550.0)")
    
    # 4. Test Visibility
    is_visible = engine.check_visibility(p_hcm, p_sat)
    print(f"Is Satellite Visible? {is_visible}")

if __name__ == "__main__":
    test_cga_basics()
