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
        
        # Disturbance on/off switch
        self.enabled = False
        
        # Position disturbance parameters (GPS errors, navigation drift)
        self.pos_noise_std = 10.0  # [m]   position noise standard deviation
        self.pos_drift_rate = 0.1  # [m/s] position drift rate
        
        # Speed disturbance parameters (wind turbulence, engine instability)
        self.spd_noise_std = 2.0   # [m/s]  speed noise standard deviation
        self.spd_drift_rate = 0.05 # [m/s²] speed drift rate
        
        # Heading disturbance parameters (control system delays, wind shear)
        self.hdg_noise_std = 2.0   # [deg]   heading noise standard deviation
        self.hdg_drift_rate = 0.1  # [deg/s] heading drift rate
        
        # Altitude disturbance parameters (atmospheric disturbance, altitude hold errors)
        self.alt_noise_std = 5.0   # [m]   altitude noise standard deviation
        self.alt_drift_rate = 0.2  # [m/s] altitude drift rate
        
        # Disturbance preset configurations
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
        
        # Per-aircraft disturbance state arrays
        with self.settrafarrays():
            # Accumulated position drift [m]
            self.pos_drift_north = np.array([])
            self.pos_drift_east = np.array([])
            
            # Accumulated speed drift [m/s]
            self.spd_drift = np.array([])
            
            # Accumulated heading drift [deg]
            self.hdg_drift = np.array([])
            
            # Accumulated altitude drift [m]
            self.alt_drift = np.array([])
    
    def create(self, n=1):
        """ Initialize disturbance state for newly created aircraft. """
        super().create(n)
        # Reset disturbance to zero for new aircraft
        self.pos_drift_north[-n:] = 0.0
        self.pos_drift_east[-n:] = 0.0
        self.spd_drift[-n:] = 0.0
        self.hdg_drift[-n:] = 0.0
        self.alt_drift[-n:] = 0.0
    
    def reset(self):
        """ Reset the disturbance system. """
        super().reset()
        self.enabled = False
    
    def set_preset(self, preset_name: str):
        """
        Apply a named disturbance preset.

        Args:
            preset_name: one of 'none', 'light', 'medium', 'heavy'

        Returns:
            (success: bool, message: str)
        """
        if preset_name not in self.presets:
            return False, f"Unknown preset: {preset_name}. Available: {list(self.presets.keys())}"

        preset = self.presets[preset_name]
        self.pos_noise_std = preset['pos_noise_std']
        self.pos_drift_rate = preset['pos_drift_rate']
        self.spd_noise_std = preset['spd_noise_std']
        self.spd_drift_rate = preset['spd_drift_rate']
        self.hdg_noise_std = preset['hdg_noise_std']
        self.hdg_drift_rate = preset['hdg_drift_rate']
        self.alt_noise_std = preset['alt_noise_std']
        self.alt_drift_rate = preset['alt_drift_rate']

        # 'none' preset disables disturbance; all others enable it
        self.enabled = (preset_name != 'none')

        # Reset accumulated drift for all aircraft
        self.pos_drift_north[:] = 0.0
        self.pos_drift_east[:] = 0.0
        self.spd_drift[:] = 0.0
        self.hdg_drift[:] = 0.0
        self.alt_drift[:] = 0.0

        return True, f"Disturbance preset set to '{preset_name}' ({'enabled' if self.enabled else 'disabled'})"

    # ------------------------------------------------------------------
    # Natural distribution log-probability (required for importance sampling)
    # ------------------------------------------------------------------
    def natural_log_prob(self, dnorth=0.0, deast=0.0,
                         dspd=0.0, dhdg=0.0, dalt=0.0):
        """Compute the log probability density of the given disturbance vector.

        Uses Gaussian white-noise terms as an approximation (accumulated drift
        is treated as state rather than action). When disturbance is disabled
        or the corresponding standard deviation is 0, the term contributes 0.

        Args:
            dnorth, deast: position disturbance [m]
            dspd: speed disturbance [m/s]
            dhdg: heading disturbance [deg]
            dalt: altitude disturbance [m]

        Returns:
            float: total log probability density
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
        Compute per-aircraft position disturbance.
        
        Returns:
            dnorth, deast: north and east position offsets [m]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf), np.zeros(bs.traf.ntraf)
        
        # Gaussian white noise
        noise_north = np.random.normal(0, self.pos_noise_std, bs.traf.ntraf)
        noise_east = np.random.normal(0, self.pos_noise_std, bs.traf.ntraf)
        
        # Accumulated drift (random walk)
        drift_north = np.random.normal(0, self.pos_drift_rate * dt, bs.traf.ntraf)
        drift_east = np.random.normal(0, self.pos_drift_rate * dt, bs.traf.ntraf)
        
        self.pos_drift_north += drift_north
        self.pos_drift_east += drift_east
        
        # Total disturbance = white noise + accumulated drift
        dnorth = noise_north + self.pos_drift_north
        deast = noise_east + self.pos_drift_east
        
        return dnorth, deast
    
    def get_speed_disturbance(self, dt):
        """
        Compute per-aircraft speed disturbance.
        
        Returns:
            dspd: speed offset [m/s]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf)
        
        # Gaussian white noise
        noise = np.random.normal(0, self.spd_noise_std, bs.traf.ntraf)
        
        # Accumulated drift
        drift = np.random.normal(0, self.spd_drift_rate * dt, bs.traf.ntraf)
        self.spd_drift += drift
        
        # Clamp accumulated drift to prevent unbounded growth
        max_drift = self.spd_noise_std * 3
        self.spd_drift = np.clip(self.spd_drift, -max_drift, max_drift)
        
        # Total disturbance
        dspd = noise + self.spd_drift
        
        return dspd
    
    def get_heading_disturbance(self, dt):
        """
        Compute per-aircraft heading disturbance.
        
        Returns:
            dhdg: heading offset [deg]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf)
        
        # Gaussian white noise
        noise = np.random.normal(0, self.hdg_noise_std, bs.traf.ntraf)
        
        # Accumulated drift
        drift = np.random.normal(0, self.hdg_drift_rate * dt, bs.traf.ntraf)
        self.hdg_drift += drift
        
        # Clamp accumulated drift
        max_drift = self.hdg_noise_std * 5
        self.hdg_drift = np.clip(self.hdg_drift, -max_drift, max_drift)
        
        # Total disturbance
        dhdg = noise + self.hdg_drift
        
        return dhdg
    
    def get_altitude_disturbance(self, dt):
        """
        Compute per-aircraft altitude disturbance.
        
        Returns:
            dalt: altitude offset [m]
        """
        if not self.enabled or bs.traf.ntraf == 0:
            return np.zeros(bs.traf.ntraf)
        
        # Gaussian white noise
        noise = np.random.normal(0, self.alt_noise_std, bs.traf.ntraf)
        
        # Accumulated drift
        drift = np.random.normal(0, self.alt_drift_rate * dt, bs.traf.ntraf)
        self.alt_drift += drift
        
        # Clamp accumulated drift
        max_drift = self.alt_noise_std * 4
        self.alt_drift = np.clip(self.alt_drift, -max_drift, max_drift)
        
        # Total disturbance
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
