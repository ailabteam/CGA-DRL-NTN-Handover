# src/core/cga_math.py
import numpy as np
import math
from clifford.g3c import *
from clifford.tools.g3c import *
# XÓA DÒNG NÀY: from clifford import inter_e_mt

# Hằng số vật lý (Đơn vị: km)
R_EARTH = 6371.0 

class CGAEngine:
    """
    Engine xử lý toán học Conformal Geometric Algebra cho bài toán Vệ tinh.
    Hệ cơ sở G4,1: e1, e2, e3, e4, e5.
    Mapping: e_inf (infinity), e_o (origin).
    """
    def __init__(self):
        # Cache các biến cơ sở để tăng tốc
        self.e_inf = einf
        self.e_o = eo
        
        # Biểu diễn Trái Đất (Sphere tại gốc O, bán kính R)
        # S = e_o - 0.5 * R^2 * e_inf
        self.earth_sphere = self.e_o - 0.5 * (R_EARTH**2) * self.e_inf

    def latlon_to_cga(self, lat, lon, alt=0.0):
        """
        Chuyển đổi Kinh độ/Vĩ độ/Độ cao -> CGA Point (Multivector).
        """
        r = R_EARTH + alt
        phi = np.radians(lat)
        theta = np.radians(lon)

        # Chuyển sang Cartesian
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        
        # Tạo vector 3D GA
        x_vec = x*e1 + y*e2 + z*e3
        
        # Up-projection: P = x + 0.5*x^2*e_inf + e_o
        # Hàm up() được import từ clifford.tools.g3c
        return up(x_vec)

    def get_distance_sq(self, P1, P2):
        """
        Tính khoảng cách Euclide bình phương giữa 2 điểm CGA.
        Công thức: P1 . P2 = -0.5 * d^2
        """
        # Toán tử | là inner product
        inner_prod = P1 | P2 
        dist_sq = -2.0 * float(inner_prod)
        return abs(dist_sq) 

    def propagate_satellite(self, sat_point, angular_velocity_rad, dt, orbit_normal_bivector):
        """
        Di chuyển vệ tinh bằng Rotor: P_new = R * P * ~R
        """
        angle = angular_velocity_rad * dt
        
        # Rotor R = exp(-B * angle / 2)
        # Sử dụng hàm exp từ math cho số thực, nhưng đây là multivector
        # Nên dùng phương thức .exp() của multivector clifford hoặc chuỗi Taylor
        # Cách chuẩn trong clifford:
        rotor = math.e**(-0.5 * angle * orbit_normal_bivector)
        
        # Update vị trí
        new_pos = rotor * sat_point * ~rotor
        return new_pos

    def check_visibility_fast(self, user_point, sat_point, min_elevation_deg=10.0):
        """
        Kiểm tra Line-of-Sight (LoS) nhanh.
        """
        # Trích xuất vector 3D (Grade 1 components e1, e2, e3)
        # down() convert CGA point -> 3D vector GA
        # Nhưng để an toàn và nhanh, ta lấy hệ số trực tiếp
        
        u_vec_ga = user_point(1) 
        s_vec_ga = sat_point(1)
        
        # Lấy hệ số vô hướng của e1, e2, e3
        u_arr = np.array([float(u_vec_ga | e1), float(u_vec_ga | e2), float(u_vec_ga | e3)])
        s_arr = np.array([float(s_vec_ga | e1), float(s_vec_ga | e2), float(s_vec_ga | e3)])
        
        # Norm
        r_u = np.linalg.norm(u_arr)
        r_s = np.linalg.norm(s_arr)
        
        if r_u == 0 or r_s == 0: return False

        # Cos(gamma) = (u . s) / (|u| |s|)
        cos_gamma = np.dot(u_arr, s_arr) / (r_u * r_s)
        
        # Điều kiện Horizon: cos(gamma) > R_E / R_S
        # horizon_threshold = R_EARTH / r_s
        # Tuy nhiên để tính cả elevation angle:
        # gamma < arccos(R/r) - epsilon
        
        gamma_rad = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        horizon_gamma = np.arccos(R_EARTH / r_s)
        
        min_el_rad = np.radians(min_elevation_deg)
        
        # Heuristic check
        if gamma_rad < (horizon_gamma - min_el_rad/4.0): 
             return True
        return False

    def to_features(self, user_point, sat_point):
        """
        Tạo vector đặc trưng (State) cho DRL.
        """
        d_sq = self.get_distance_sq(user_point, sat_point)
        dist = np.sqrt(d_sq)
        
        # Feature 1: Khoảng cách (Chuẩn hóa max 2000km)
        feat_dist = dist / 2000.0
        
        # Feature 2: Cosine góc kẹp
        u_vec_ga = user_point(1) 
        s_vec_ga = sat_point(1)
        u_arr = np.array([float(u_vec_ga | e1), float(u_vec_ga | e2), float(u_vec_ga | e3)])
        s_arr = np.array([float(s_vec_ga | e1), float(s_vec_ga | e2), float(s_vec_ga | e3)])
        
        norm_prod = (np.linalg.norm(u_arr) * np.linalg.norm(s_arr))
        if norm_prod == 0:
            cos_gamma = 0.0
        else:
            cos_gamma = np.dot(u_arr, s_arr) / norm_prod
        
        return np.array([feat_dist, cos_gamma], dtype=np.float32)
