"""
No-fly zone module for BlueSky traffic simulation.

Provides functionality to define and check violations of no-fly zones,
including circular and polygonal shapes with altitude constraints.
"""

import numpy as np
from bluesky.core import Entity
from bluesky.tools.geo import kwikdist


class NoFlyZone(Entity, replaceable=True):
    """
    No-fly zone management for BlueSky simulation.
    
    Supports:
    - Circular no-fly zones (lat, lon, radius)
    - Polygonal no-fly zones (list of lat/lon vertices)
    - Altitude constraints (min/max altitude)
    - Violation detection and tracking
    """
    
    def __init__(self):
        super().__init__()
        
        # No-fly zone storage
        self.zones = []  # List of zone dictionaries
        self.zone_names = []  # List of zone names
        
        # Violation tracking
        self.violations = {}  # acid -> list of violated zone indices
        self.violation_count = {}  # acid -> total violation count
        
    def create_circular_zone(self, name, lat, lon, radius, alt_min=0, alt_max=99999):
        """
        Create a circular no-fly zone.
        
        Arguments:
        - name: Unique identifier for the zone
        - lat: Latitude of center (degrees)
        - lon: Longitude of center (degrees)
        - radius: Radius (NM)
        - alt_min: Minimum altitude (m), default 0
        - alt_max: Maximum altitude (m), default 99999
        
        Returns:
        - Zone index
        """
        zone = {
            'name': name,
            'type': 'circle',
            'lat': lat,
            'lon': lon,
            'radius': radius,  # NM
            'alt_min': alt_min,
            'alt_max': alt_max
        }
        
        self.zones.append(zone)
        self.zone_names.append(name)
        
        return len(self.zones) - 1
    
    def create_polygon_zone(self, name, lats, lons, alt_min=0, alt_max=99999):
        """
        Create a polygonal no-fly zone.
        
        Arguments:
        - name: Unique identifier for the zone
        - lats: List of vertex latitudes (degrees)
        - lons: List of vertex longitudes (degrees)
        - alt_min: Minimum altitude (m), default 0
        - alt_max: Maximum altitude (m), default 99999
        
        Returns:
        - Zone index
        """
        if len(lats) < 3 or len(lons) < 3:
            raise ValueError("Polygon zone must have at least 3 vertices")
        
        if len(lats) != len(lons):
            raise ValueError("Number of latitudes must equal number of longitudes")
        
        zone = {
            'name': name,
            'type': 'polygon',
            'lats': np.array(lats),
            'lons': np.array(lons),
            'alt_min': alt_min,
            'alt_max': alt_max
        }
        
        self.zones.append(zone)
        self.zone_names.append(name)
        
        return len(self.zones) - 1
    
    def delete_zone(self, name):
        """
        Delete a no-fly zone by name.
        
        Arguments:
        - name: Zone name to delete
        
        Returns:
        - True if deleted, False if not found
        """
        if name in self.zone_names:
            idx = self.zone_names.index(name)
            del self.zones[idx]
            del self.zone_names[idx]
            return True
        return False
    
    def clear_zones(self):
        """ Clear all no-fly zones """
        self.zones = []
        self.zone_names = []
        self.violations = {}
        self.violation_count = {}
    
    def check_point(self, lat, lon, alt):
        """
        Check if a point violates any no-fly zone.
        
        Arguments:
        - lat: Latitude (degrees)
        - lon: Longitude (degrees)
        - alt: Altitude (m)
        
        Returns:
        - List of violated zone indices (empty if no violations)
        """
        violated_zones = []
        
        for i, zone in enumerate(self.zones):
            # Check altitude constraint first (quick rejection)
            if alt < zone['alt_min'] or alt > zone['alt_max']:
                continue
            
            # Check horizontal constraint
            if zone['type'] == 'circle':
                if self._point_in_circle(lat, lon, zone):
                    violated_zones.append(i)
            
            elif zone['type'] == 'polygon':
                if self._point_in_polygon(lat, lon, zone):
                    violated_zones.append(i)
        
        return violated_zones
    
    def check_aircraft(self, acid, lat, lon, alt):
        """
        Check if an aircraft violates any no-fly zone and update tracking.
        
        Arguments:
        - acid: Aircraft ID
        - lat: Latitude (degrees)
        - lon: Longitude (degrees)
        - alt: Altitude (m)
        
        Returns:
        - List of violated zone indices
        """
        violated_zones = self.check_point(lat, lon, alt)
        
        # Update violation tracking
        if violated_zones:
            if acid not in self.violations:
                self.violations[acid] = []
                self.violation_count[acid] = 0
            
            # Track new violations
            for zone_idx in violated_zones:
                if zone_idx not in self.violations[acid]:
                    self.violations[acid].append(zone_idx)
                    self.violation_count[acid] += 1
        else:
            # Clear current violations if aircraft exits all zones
            if acid in self.violations:
                self.violations[acid] = []
        
        return violated_zones
    
    def get_violations(self, acid=None):
        """
        Get violation information for aircraft.
        
        Arguments:
        - acid: Aircraft ID (if None, return all violations)
        
        Returns:
        - Dictionary of violations or list for specific aircraft
        """
        if acid is None:
            return self.violations
        return self.violations.get(acid, [])
    
    def get_violation_count(self, acid=None):
        """
        Get total violation count for aircraft.
        
        Arguments:
        - acid: Aircraft ID (if None, return all counts)
        
        Returns:
        - Total violation count
        """
        if acid is None:
            return self.violation_count
        return self.violation_count.get(acid, 0)
    
    def _point_in_circle(self, lat, lon, zone):
        """
        Check if point is inside circular zone.
        
        Arguments:
        - lat: Point latitude
        - lon: Point longitude
        - zone: Zone dictionary with 'lat', 'lon', 'radius'
        
        Returns:
        - True if inside circle
        """
        dist = kwikdist(zone['lat'], zone['lon'], lat, lon)  # Returns distance in NM
        return dist < zone['radius']
    
    def _point_in_polygon(self, lat, lon, zone):
        """
        Check if point is inside polygonal zone using ray casting algorithm.
        
        Arguments:
        - lat: Point latitude
        - lon: Point longitude
        - zone: Zone dictionary with 'lats', 'lons' arrays
        
        Returns:
        - True if inside polygon
        """
        lats = zone['lats']
        lons = zone['lons']
        n = len(lats)
        
        inside = False
        j = n - 1
        
        for i in range(n):
            # Check if point is on an edge or vertex
            if ((lats[i] > lat) != (lats[j] > lat)) and \
               (lon < (lons[j] - lons[i]) * (lat - lats[i]) / (lats[j] - lats[i]) + lons[i]):
                inside = not inside
            j = i
        
        return inside
    
    def get_zone_info(self, name=None):
        """
        Get information about no-fly zones.
        
        Arguments:
        - name: Zone name (if None, return info for all zones)
        
        Returns:
        - Zone information string
        """
        if name is None:
            # Return info for all zones
            if not self.zones:
                return "No no-fly zones defined"
            
            info = "No-Fly Zones:\n"
            info += "=" * 60 + "\n"
            
            for i, zone in enumerate(self.zones):
                info += f"\n{i+1}. {zone['name']} ({zone['type']})\n"
                
                if zone['type'] == 'circle':
                    info += f"   Center: ({zone['lat']:.4f}°, {zone['lon']:.4f}°)\n"
                    info += f"   Radius: {zone['radius']:.2f} NM\n"
                elif zone['type'] == 'polygon':
                    info += f"   Vertices: {len(zone['lats'])}\n"
                    info += f"   Bounds: ({np.min(zone['lats']):.4f}° to {np.max(zone['lats']):.4f}°, "
                    info += f"{np.min(zone['lons']):.4f}° to {np.max(zone['lons']):.4f}°)\n"
                
                info += f"   Altitude: {zone['alt_min']:.0f}m to {zone['alt_max']:.0f}m\n"
            
            return info
        else:
            # Return info for specific zone
            if name in self.zone_names:
                idx = self.zone_names.index(name)
                zone = self.zones[idx]
                
                info = f"Zone: {zone['name']} ({zone['type']})\n"
                
                if zone['type'] == 'circle':
                    info += f"Center: ({zone['lat']:.4f}°, {zone['lon']:.4f}°)\n"
                    info += f"Radius: {zone['radius']:.2f} NM\n"
                elif zone['type'] == 'polygon':
                    info += f"Vertices:\n"
                    for lat, lon in zip(zone['lats'], zone['lons']):
                        info += f"  ({lat:.4f}°, {lon:.4f}°)\n"
                
                info += f"Altitude: {zone['alt_min']:.0f}m to {zone['alt_max']:.0f}m"
                
                return info
            else:
                return f"Zone '{name}' not found"
    
    def info(self):
        """ Get general no-fly zone information """
        return self.get_zone_info()
