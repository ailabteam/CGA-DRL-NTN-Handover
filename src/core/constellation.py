# src/core/constellation.py
import numpy as np
from src.core.cga_math import CGAEngine
from clifford.g3c import e1, e2, e3

class WalkerConstellation:
    """
    Trình tạo chùm vệ tinh theo cấu trúc Walker Delta.
    """
    def __init__(self, engine: CGAEngine, total_sats=66, n_planes=6, inclination=86.4, altitude=780.0):
        self.engine = engine
        self.total_sats = total_sats
        self.n_planes = n_planes
        self.sats_per_plane = total_sats // n_planes
        self.inclination = inclination
        self.altitude = altitude
        
        # GM trái đất ~ 3.986e5 km^3/s^2
        GM = 3.986004418e5
        r = 6371.0 + altitude
        self.omega = np.sqrt(GM / (r**3)) # rad/s
        
        self.satellites = []
        self._generate_walker()

    def _generate_walker(self):
        self.satellites = []
        
        # Điểm gốc tại xích đạo
        start_point = self.engine.latlon_to_cga(0.0, 0.0, self.altitude)
        
        # Vector 0 dùng cho khởi tạo velocity ban đầu
        zero_vec = 0.0 * e1 

        for p in range(self.n_planes):
            raan = (360.0 / self.n_planes) * p
            
            # Rotor RAAN (quanh Z)
            B_xy = e1^e2
            R_node = np.e**(-0.5 * np.radians(raan) * B_xy)
            
            # Rotor Inclination (quanh trục X' sau khi xoay node)
            # Giả sử nghiêng quanh e2^e3
            B_yz = e2^e3
            R_inc = np.e**(-0.5 * np.radians(self.inclination) * B_yz)
            
            R_plane = R_node * R_inc
            
            # Pháp tuyến quỹ đạo
            orbit_normal = R_plane * (e1^e2) * ~R_plane
            
            for s in range(self.sats_per_plane):
                mean_anomaly = (360.0 / self.sats_per_plane) * s
                
                # Rotor vị trí trên quỹ đạo
                R_sat = np.e**(-0.5 * np.radians(mean_anomaly) * (e1^e2))
                
                pos_local = R_sat * start_point * ~R_sat
                pos_final = R_plane * pos_local * ~R_plane
                
                sat_id = p * self.sats_per_plane + s
                
                self.satellites.append({
                    'id': sat_id,
                    'pos': pos_final,
                    'plane_normal': orbit_normal,
                    'velocity': zero_vec # Khởi tạo
                })

    def propagate(self, dt):
        """
        Di chuyển vệ tinh và tính vận tốc tức thời.
        """
        for sat in self.satellites:
            old_pos_vec = sat['pos'](1) # Lấy vector cũ
            
            # Di chuyển
            sat['pos'] = self.engine.propagate_satellite(
                sat['pos'], 
                self.omega, 
                dt, 
                sat['plane_normal']
            )
            
            # Tính vector vận tốc (Linear Velocity Vector)
            # v = (p_new - p_old) / dt
            new_pos_vec = sat['pos'](1)
            sat['velocity'] = (new_pos_vec - old_pos_vec) / dt
