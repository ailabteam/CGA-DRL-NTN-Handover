# test_math.py
import numpy as np
from src.core.cga_math import CGAEngine, R_EARTH
from clifford.g3c import e1, e2

def run_tests():
    print("=== STARTING CGA MATH VALIDATION ===")
    engine = CGAEngine()
    
    # TEST 1: Mapping tọa độ
    print("\n[Test 1] Coordinate Mapping (Lat/Lon -> CGA)")
    lat, lon = 10.762, 106.660 # HCM City
    p_hcm = engine.latlon_to_cga(lat, lon, alt=0.0)
    print(f"   -> Point created successfully.")
    
    # TEST 2: Khoảng cách địa lý
    print("\n[Test 2] Distance Calculation (HCM -> Hanoi)")
    lat_hn, lon_hn = 21.028, 105.854
    p_hn = engine.latlon_to_cga(lat_hn, lon_hn, alt=0.0)
    
    d_sq = engine.get_distance_sq(p_hcm, p_hn)
    d_km = np.sqrt(d_sq)
    print(f"   -> Calculated Distance: {d_km:.2f} km")
    
    if 1100 < d_km < 1200:
        print("   -> [PASS] Distance is reasonable (Chord length approx 1137km).")
    else:
        print("   -> [WARNING] Distance seems off!")

    # TEST 3: Vệ tinh & Visibility
    print("\n[Test 3] Satellite Visibility Check")
    # Giả sử vệ tinh ở độ cao 550km ngay trên đỉnh đầu HCM
    p_sat_overhead = engine.latlon_to_cga(lat, lon, alt=550.0)
    vis_overhead = engine.check_visibility_fast(p_hcm, p_sat_overhead)
    print(f"   -> Overhead (Zenith): {vis_overhead} (Expect True)")
    
    # Giả sử vệ tinh ở phía bên kia trái đất (Lon + 180)
    p_sat_far = engine.latlon_to_cga(lat, lon + 180.0, alt=550.0)
    vis_far = engine.check_visibility_fast(p_hcm, p_sat_far)
    print(f"   -> Other side of Earth: {vis_far} (Expect False)")
    
    # TEST 4: Rotor Movement
    print("\n[Test 4] Rotor Propagation")
    # Quay vệ tinh 90 độ quanh trục Z (e1^e2)
    p_sat_start = engine.latlon_to_cga(0, 0, 550.0) # Tại xích đạo, kinh tuyến 0
    bivector_z = e1^e2 # Mặt phẳng XY (Xích đạo)
    
    # Quay 90 độ (pi/2)
    omega = 1.0 # rad/s
    dt = np.pi / 2 # s
    p_sat_moved = engine.propagate_satellite(p_sat_start, omega, dt, bivector_z)
    
    # Kiểm tra vị trí mới (Nên ở kinh độ 90)
    # Lấy tọa độ x, y
    x_new = float(p_sat_moved(1) | e1)
    y_new = float(p_sat_moved(1) | e2)
    print(f"   -> New Pos (x, y): ({x_new:.1f}, {y_new:.1f})")
    
    if abs(x_new) < 1.0 and y_new > 6000:
        print("   -> [PASS] Satellite rotated 90 degrees correctly.")
    else:
        print("   -> [FAIL] Rotation logic error.")

if __name__ == "__main__":
    run_tests()
