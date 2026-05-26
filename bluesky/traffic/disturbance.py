""" Natural disturbance module for BlueSky.

Adds realistic disturbance effects to aircraft including:
- Position disturbance: GPS errors, navigation drift
- Speed disturbance: Wind turbulence, engine instability  
- Heading disturbance: Control system delays, wind shear
- Altitude disturbance: Atmospheric disturbance, altitude hold errors
"""

import numpy as np
import bluesky as bs
from bluesky.core import Entity


class Disturbance(Entity, replaceable=True):
    """ Natural disturbance model for adding realistic noise to aircraft states. """
    
    def __init__(self):
        super().__init__()
        
        # 扰动开关
        self.enabled = False
        
        # 位置扰动参数 (GPS误差、导航偏差)
        self.pos_noise_std = 10.0  # [m] 位置噪声标准差
        self.pos_drift_rate = 0.1  # [m/s] 位置漂移速率
        
        # 速度扰动参数 (风扰动、发动机不稳定)
        self.spd_noise_std = 2.0   # [m/s] 速度噪声标准差
        self.spd_drift_rate = 0.05 # [m/s^2] 速度漂移率
        
        # 航向扰动参数 (控制系统延迟、风切变)
        self.hdg_noise_std = 2.0   # [deg] 航向噪声标准差
        self.hdg_drift_rate = 0.1  # [deg/s] 航向漂移速率
        
        # 高度扰动参数 (大气扰动、高度保持误差)
        self.alt_noise_std = 5.0   # [m] 高度噪声标准差
        self.alt_drift_rate = 0.2  # [m/s] 高度漂移速率
        
        # 扰动预设配置
        self.presets = {
            'none': {
                'pos_noise_std': 0.0,
                'pos_drift_rate': 0.0,
                'spd_noise_std': 0.0,
                'spd_drift_rate': 0.0,
                'hdg_noise_std': 0.0,
                'hdg_drift_rate': 0.0,
                'alt_noise_std': 0.0,
                'alt_drift_rate': 0.0
            },
            'light': {
                'pos_noise_std': 5.0,
                'pos_drift_rate': 0.05,
                'spd_noise_std': 1.0,
                'spd_drift_rate': 0.02,
                'hdg_noise_std': 1.0,
                'hdg_drift_rate': 0.05,
                'alt_noise_std': 3.0,
                'alt_drift_rate': 0.1
            },
            'medium': {
                'pos_noise_std': 10.0,
                'pos_drift_rate': 0.1,
                'spd_noise_std': 2.0,
                'spd_drift_rate': 0.05,
                'hdg_noise_std': 2.0,
                'hdg_drift_rate': 0.1,
                'alt_noise_std': 5.0,
                'alt_drift_rate': 0.2
            },
            'heavy': {
                'pos_noise_std': 20.0,
                'pos_drift_rate': 0.2,
                'spd_noise_std': 4.0,
                'spd_drift_rate': 0.1,
                'hdg_noise_std': 4.0,
                'hdg_drift_rate': 0.2,
                'alt_noise_std': 10.0,
                'alt_drift_rate': 0.4
            }
        }
        
        # 每架飞机的扰动状态
        with self.settrafarrays():
            # 位置累积漂移 [m]
            self.pos_drift_north = np.array([])
            self.pos_drift_east = np.array([])
            
            # 速度累积漂移 [m/s]
            self.spd_drift = np.array([])
            
            # 航向累积漂移 [deg]
            self.hdg_drift = np.array([])
            
            # 高度累积漂移 [m]
            self.alt_drift = np.array([])
    
    def create(self, n=1):
        """创建新飞行器时初始化扰动状态"""
        super().create(n)
        # 初始化扰动为0
        self.pos_drift_north[-n:] = 0.0
        self.pos_drift_east[-n:] = 0.0
        self.spd_drift[-n:] = 0.0
        self.hdg_drift[-n:] = 0.0
        self.alt_drift[-n:] = 0.0
    
    def reset(self):
        """重置扰动系统"""
        super().reset()
        self.enabled = False
    
    def set_preset(self, preset_name: str):
        """
        设置扰动预设配置

        参数:
            preset_name: 预设名称 ('none', 'light', 'medium', 'heavy')

        返回:
            (success, message)
        """
        if preset_name not in self.presets:
            return False, f"未知预设: {preset_name}. 可用: {list(self.presets.keys())}"

        preset = self.presets[preset_name]
        self.pos_noise_std = preset['pos_noise_std']
        self.pos_drift_rate = preset['pos_drift_rate']
        self.spd_noise_std = preset['spd_noise_std']
        self.spd_drift_rate = preset['spd_drift_rate']
        self.hdg_noise_std = preset['hdg_noise_std']
        self.hdg_drift_rate = preset['hdg_drift_rate']
        self.alt_noise_std = preset['alt_noise_std']
        self.alt_drift_rate = preset['alt_drift_rate']

        # 'none' 预设关闭扰动；其他预设开启
        self.enabled = (preset_name != 'none')

        # 重置所有飞行器的累积漂移
        self.pos_drift_north[:] = 0.0
        self.pos_drift_east[:] = 0.0
        self.spd_drift[:] = 0.0
        self.hdg_drift[:] = 0.0
        self.alt_drift[:] = 0.0

        return True, f"已设置扰动预设: {preset_name} ({'enabled' if self.enabled else 'disabled'})"

    # ------------------------------------------------------------------
    # 自然分布概率密度（重要性采样所需）
    # ------------------------------------------------------------------
    def natural_log_prob(self, dnorth=0.0, deast=0.0,
                         dspd=0.0, dhdg=0.0, dalt=0.0):
        """计算给定扰动量在当前自然分布下的对数概率密度。

        以高斯白噪声项的密度作为近似（累积漂移项视为状态而非动作）。
        当扰动关闭或对应方差为 0 时，相应项概率为 1（log_prob=0）。

        参数:
            dnorth, deast: 位置扰动 [m]
            dspd: 速度扰动 [m/s]
            dhdg: 航向扰动 [deg]
            dalt: 高度扰动 [m]

        返回:
            float: 总对数概率密度
        """
        log_p = 0.0
        if self.enabled:
            for value, std in (
                (dnorth, self.pos_noise_std),
                (deast,  self.pos_noise_std),
                (dspd,   self.spd_noise_std),
                (dhdg,   self.hdg_noise_std),
                (dalt,   self.alt_noise_std),
            ):
                if std > 1e-9:
                    log_p += (-0.5 * (value / std) ** 2
                              - 0.5 * np.log(2 * np.pi * std * std))
        return float(log_p)
    
    def get_position_disturbance(self, dt):
        """
        计算位置扰动
        
        返回:
            dnorth, deast: 北向和东向的位置偏移 [m]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf), np.zeros(bs.traf.ntraf)
        
        # 高斯白噪声
        noise_north = np.random.normal(0, self.pos_noise_std, bs.traf.ntraf)
        noise_east = np.random.normal(0, self.pos_noise_std, bs.traf.ntraf)
        
        # 累积漂移 (随机游走)
        drift_north = np.random.normal(0, self.pos_drift_rate * dt, bs.traf.ntraf)
        drift_east = np.random.normal(0, self.pos_drift_rate * dt, bs.traf.ntraf)
        
        self.pos_drift_north += drift_north
        self.pos_drift_east += drift_east
        
        # 总扰动 = 白噪声 + 累积漂移
        dnorth = noise_north + self.pos_drift_north
        deast = noise_east + self.pos_drift_east
        
        return dnorth, deast
    
    def get_speed_disturbance(self, dt):
        """
        计算速度扰动
        
        返回:
            dspd: 速度偏移 [m/s]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf)
        
        # 高斯白噪声
        noise = np.random.normal(0, self.spd_noise_std, bs.traf.ntraf)
        
        # 累积漂移
        drift = np.random.normal(0, self.spd_drift_rate * dt, bs.traf.ntraf)
        self.spd_drift += drift
        
        # 限制累积漂移幅度（避免无限增长）
        max_drift = self.spd_noise_std * 3
        self.spd_drift = np.clip(self.spd_drift, -max_drift, max_drift)
        
        # 总扰动
        dspd = noise + self.spd_drift
        
        return dspd
    
    def get_heading_disturbance(self, dt):
        """
        计算航向扰动
        
        返回:
            dhdg: 航向偏移 [deg]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf)
        
        # 高斯白噪声
        noise = np.random.normal(0, self.hdg_noise_std, bs.traf.ntraf)
        
        # 累积漂移
        drift = np.random.normal(0, self.hdg_drift_rate * dt, bs.traf.ntraf)
        self.hdg_drift += drift
        
        # 限制累积漂移幅度
        max_drift = self.hdg_noise_std * 5
        self.hdg_drift = np.clip(self.hdg_drift, -max_drift, max_drift)
        
        # 总扰动
        dhdg = noise + self.hdg_drift
        
        return dhdg
    
    def get_altitude_disturbance(self, dt):
        """
        计算高度扰动
        
        返回:
            dalt: 高度偏移 [m]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf)
        
        # 高斯白噪声
        noise = np.random.normal(0, self.alt_noise_std, bs.traf.ntraf)
        
        # 累积漂移
        drift = np.random.normal(0, self.alt_drift_rate * dt, bs.traf.ntraf)
        self.alt_drift += drift
        
        # 限制累积漂移幅度
        max_drift = self.alt_noise_std * 4
        self.alt_drift = np.clip(self.alt_drift, -max_drift, max_drift)
        
        # 总扰动
        dalt = noise + self.alt_drift
        
        return dalt
    
    def info(self):
        """ Get current disturbance configuration. """
        info = f"""
Natural Disturbance Configuration:
  Status: {'Enabled' if self.enabled else 'Disabled'}
  Position noise: {self.pos_noise_std:.1f} m
  Position drift: {self.pos_drift_rate:.3f} m/s
  Speed noise: {self.spd_noise_std:.1f} m/s
  Speed drift: {self.spd_drift_rate:.3f} m/s²
  Heading noise: {self.hdg_noise_std:.1f}°
  Heading drift: {self.hdg_drift_rate:.3f}°/s
  Altitude noise: {self.alt_noise_std:.1f} m
  Altitude drift: {self.alt_drift_rate:.3f} m/s
"""
        return info
