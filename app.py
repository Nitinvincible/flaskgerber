#!/usr/bin/env python3
"""
Updated PCB File Preview Flask Application with Enhanced Layer Detection
Supports comprehensive layer type identification for various PCB manufacturers
"""

import os
import re
import json
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {
    'gbr', 'ger', 'gtl', 'gbl', 'gts', 'gbs', 'gto', 'gbo', 'gtp', 'gbp',
    'drl', 'xln', 'txt', 'cnc', 'nc', 'tap', 'DRL',  # Drill files
    'pnp', 'pos', 'xy',  # Pick and place
    'zip', 'rar',  # Archives
    'pcb', 'sch',  # PCB design files
    'pdf', 'svg', 'png', 'jpg', 'jpeg',  # Documentation
    'csv', 'net', 'bom'  # BOM and netlists
}

@dataclass
class PCBLayer:
    """Represents a PCB layer with its properties and geometry"""
    name: str
    type: str
    filename: str
    apertures: Dict[str, Any]
    geometries: List[Dict[str, Any]]
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    description: str = ""

@dataclass
class DrillHole:
    """Represents a drill hole"""
    x: float
    y: float
    diameter: float
    plated: bool = True

class EnhancedLayerDetector:
    """Enhanced layer type detection supporting multiple naming conventions"""
    
    def __init__(self):
        # Comprehensive layer mapping patterns
        self.layer_patterns = [
            # === COPPER LAYERS ===
            # Top copper patterns
            (r'.*\.(gtl|cmp|top|f\.cu|front|copper_top|topper)$', 'copper_top', 'Top Copper'),
            (r'.*[-_\.]?(top|front)[-_\.]?(copper|cu|layer).*', 'copper_top', 'Top Copper'),
            (r'.*[-_\.]?(copper|cu)[-_\.]?(top|front|1).*', 'copper_top', 'Top Copper'),
            (r'.*[-_\.]?l1[-_\.]?(cu|copper)?.*', 'copper_top', 'Top Copper (L1)'),
            (r'.*[-_\.]?signal[-_\.]?1.*', 'copper_top', 'Top Copper (Signal 1)'),
            
            # Bottom copper patterns
            (r'.*\.(gbl|sol|bottom|b\.cu|back|copper_bottom|bottom)$', 'copper_bottom', 'Bottom Copper'),
            (r'.*[-_\.]?(bottom|back)[-_\.]?(copper|cu|layer).*', 'copper_bottom', 'Bottom Copper'),
            (r'.*[-_\.]?(copper|cu)[-_\.]?(bottom|back|2).*', 'copper_bottom', 'Bottom Copper'),
            (r'.*[-_\.]?l2[-_\.]?(cu|copper)?.*', 'copper_bottom', 'Bottom Copper (L2)'),
            (r'.*[-_\.]?signal[-_\.]?2.*', 'copper_bottom', 'Bottom Copper (Signal 2)'),
            
            # Inner copper layers
            (r'.*[-_\.]?l([3-9]|1[0-9]|2[0-9])[-_\.]?(cu|copper)?.*', 'copper_inner', 'Inner Copper'),
            (r'.*[-_\.]?in([1-9]|1[0-9]|2[0-9])[-_\.]?(cu|copper)?.*', 'copper_inner', 'Inner Copper'),
            (r'.*[-_\.]?signal[-_\.]?([3-9]|1[0-9]|2[0-9]).*', 'copper_inner', 'Inner Copper'),
            
            # === SOLDERMASK LAYERS ===
            # Top soldermask
            (r'.*\.(gts|stc|tsc|f\.mask|mask_top|topmask)$', 'soldermask_top', 'Top Soldermask'),
            (r'.*[-_\.]?(top|front)[-_\.]?(mask|solder|soldermask).*', 'soldermask_top', 'Top Soldermask'),
            (r'.*[-_\.]?(mask|solder|soldermask)[-_\.]?(top|front).*', 'soldermask_top', 'Top Soldermask'),
            (r'.*[-_\.]?sm[-_\.]?(top|t|front|f).*', 'soldermask_top', 'Top Soldermask'),
            
            # Bottom soldermask
            (r'.*\.(gbs|sts|bsc|b\.mask|mask_bottom|bottommask)$', 'soldermask_bottom', 'Bottom Soldermask'),
            (r'.*[-_\.]?(bottom|back)[-_\.]?(mask|solder|soldermask).*', 'soldermask_bottom', 'Bottom Soldermask'),
            (r'.*[-_\.]?(mask|solder|soldermask)[-_\.]?(bottom|back).*', 'soldermask_bottom', 'Bottom Soldermask'),
            (r'.*[-_\.]?sm[-_\.]?(bottom|b|back).*', 'soldermask_bottom', 'Bottom Soldermask'),
            
            # === SILKSCREEN LAYERS ===
            # Top silkscreen
            (r'.*\.(gto|plc|tsk|f\.silks|silks_top|topsilk|silk_top)$', 'silkscreen_top', 'Top Silkscreen'),
            (r'.*[-_\.]?(top|front)[-_\.]?(silk|silkscreen|legend|print).*', 'silkscreen_top', 'Top Silkscreen'),
            (r'.*[-_\.]?(silk|silkscreen|legend|print)[-_\.]?(top|front).*', 'silkscreen_top', 'Top Silkscreen'),
            (r'.*[-_\.]?ss[-_\.]?(top|t|front|f).*', 'silkscreen_top', 'Top Silkscreen'),
            
            # Bottom silkscreen
            (r'.*\.(gbo|pls|bsk|b\.silks|silks_bottom|bottomsilk|silk_bottom)$', 'silkscreen_bottom', 'Bottom Silkscreen'),
            (r'.*[-_\.]?(bottom|back)[-_\.]?(silk|silkscreen|legend|print).*', 'silkscreen_bottom', 'Bottom Silkscreen'),
            (r'.*[-_\.]?(silk|silkscreen|legend|print)[-_\.]?(bottom|back).*', 'silkscreen_bottom', 'Bottom Silkscreen'),
            (r'.*[-_\.]?ss[-_\.]?(bottom|b|back).*', 'silkscreen_bottom', 'Bottom Silkscreen'),
            
            # === PASTE LAYERS ===
            # Top paste
            (r'.*\.(gtp|f\.paste|paste_top|toppaste|pt)$', 'paste_top', 'Top Paste'),
            (r'.*[-_\.]?(top|front)[-_\.]?(paste|stencil).*', 'paste_top', 'Top Paste'),
            (r'.*[-_\.]?(paste|stencil)[-_\.]?(top|front).*', 'paste_top', 'Top Paste'),
            (r'.*[-_\.]?pastestencil[-_\.]?top.*', 'paste_top', 'Top Paste'),
            
            # Bottom paste  
            (r'.*\.(gbp|b\.paste|paste_bottom|bottompaste|pb)$', 'paste_bottom', 'Bottom Paste'),
            (r'.*[-_\.]?(bottom|back)[-_\.]?(paste|stencil).*', 'paste_bottom', 'Bottom Paste'),
            (r'.*[-_\.]?(paste|stencil)[-_\.]?(bottom|back).*', 'paste_bottom', 'Bottom Paste'),
            (r'.*[-_\.]?pastestencil[-_\.]?bottom.*', 'paste_bottom', 'Bottom Paste'),
            
            # === OUTLINE/MECHANICAL ===
            (r'.*\.(gko|gm1|out|mill|outline|edge|mechanical|board|cutout)$', 'outline', 'Board Outline'),
            (r'.*[-_\.]?(outline|edge|mill|milling|cutout|mechanical|board).*', 'outline', 'Board Outline'),
            (r'.*[-_\.]?edge[-_\.]?cuts.*', 'outline', 'Board Outline'),
            (r'.*[-_\.]?boardoutline.*', 'outline', 'Board Outline'),
            
            # === DRILL FILES ===
            (r'.*\.(drl|xln|txt|cnc|nc|tap|drill|exc)$', 'drill', 'Drill File'),
            (r'.*[-_\.]?(drill|hole|via)s?[-_\.]?.*', 'drill', 'Drill File'),
            (r'.*[-_\.]?excellon.*', 'drill', 'Drill File'),
            
            # === DOCUMENTATION ===
            (r'.*\.(pdf|svg|png|jpg|jpeg|bmp|tiff?)$', 'document', 'Documentation'),
            (r'.*[-_\.]?(doc|documentation|readme|notes).*', 'document', 'Documentation'),
            (r'.*[-_\.]?(fab|fabrication)[-_\.]?(notes?|drawing).*', 'document', 'Fabrication Notes'),
            (r'.*[-_\.]?assembly[-_\.]?(notes?|drawing).*', 'document', 'Assembly Notes'),
            
            # === PICK AND PLACE ===
            (r'.*\.(pnp|pos|xy|place|pick|component)$', 'document', 'Pick and Place'),
            (r'.*[-_\.]?(pick|place|placement|component|pos|position).*', 'document', 'Pick and Place'),
            
            # === NETLIST AND BOM ===
            (r'.*\.(net|netlist|bom|csv)$', 'document', 'Netlist/BOM'),
            (r'.*[-_\.]?(netlist|bom|bill|parts?)[-_\.]?.*', 'document', 'Netlist/BOM'),
        ]
        
        # Specific filename mappings for common manufacturers
        self.exact_mappings = {
            # JLCPCB common names
            'gerber_job': 'document',
            'gcode': 'drill',
            'report': 'document',
            
            # EasyEDA exports
            'copper_bottom.gbr': 'copper_bottom',
            'copper_top.gbr': 'copper_top',
            'soldermask_bottom.gbr': 'soldermask_bottom',
            'soldermask_top.gbr': 'soldermask_top',
            'silkscreen_bottom.gbr': 'silkscreen_bottom',
            'silkscreen_top.gbr': 'silkscreen_top',
            'pastestencil_bottom.gbr': 'paste_bottom',
            'pastestencil_top.gbr': 'paste_top',
            'boardoutline.gbr': 'outline',
            
            # KiCad patterns
            'f_cu.gbr': 'copper_top',
            'b_cu.gbr': 'copper_bottom',
            'f_mask.gbr': 'soldermask_top',
            'b_mask.gbr': 'soldermask_bottom',
            'f_silks.gbr': 'silkscreen_top',
            'b_silks.gbr': 'silkscreen_bottom',
            'f_paste.gbr': 'paste_top',
            'b_paste.gbr': 'paste_bottom',
            'edge_cuts.gbr': 'outline',
        }
        
        # Priority order for layer types
        self.layer_priority = {
            'unknown': 0,
            'document': 1,
            'copper_inner': 2,
            'paste_bottom': 3,
            'paste_top': 4,
            'soldermask_bottom': 5,
            'soldermask_top': 6,
            'silkscreen_bottom': 7,
            'silkscreen_top': 8,
            'outline': 9,
            'drill': 10,
            'copper_bottom': 11,
            'copper_top': 12,
        }
    
    def detect_layer_type(self, filename: str, file_content: Optional[str] = None) -> Tuple[str, str]:
        """
        Detect layer type from filename and optionally file content
        Returns (layer_type, description)
        """
        filename_lower = filename.lower().strip()
        basename = filename_lower.split('/')[-1]  # Remove path
        
        # Check exact mappings first
        if basename in self.exact_mappings:
            layer_type = self.exact_mappings[basename]
            return layer_type, self._get_layer_description(layer_type)
        
        # Try pattern matching
        matches = []
        for pattern, layer_type, description in self.layer_patterns:
            if re.match(pattern, filename_lower, re.IGNORECASE):
                matches.append((layer_type, description, self.layer_priority.get(layer_type, 0)))
        
        # If we have matches, return the highest priority one
        if matches:
            # Sort by priority (highest first)
            matches.sort(key=lambda x: x[2], reverse=True)
            return matches[0][0], matches[0][1]
        
        # Content-based detection as fallback
        if file_content:
            content_type = self._detect_from_content(file_content)
            if content_type != 'unknown':
                return content_type, self._get_layer_description(content_type)
        
        return 'unknown', 'Unknown File Type'
    
    def _detect_from_content(self, content: str) -> str:
        """Detect layer type from file content"""
        content_lower = content.lower()
        
        # Check for Gerber format indicators
        if any(indicator in content_lower for indicator in ['%fslax', '%momm', '%moin', 'g04', '%add']):
            # This is likely a Gerber file, try to determine layer type from content
            if any(term in content_lower for term in ['copper', 'signal', 'power', 'ground']):
                return 'copper_top'  # Default to top if ambiguous
            elif any(term in content_lower for term in ['mask', 'solder']):
                return 'soldermask_top'  # Default to top if ambiguous
            elif any(term in content_lower for term in ['silk', 'legend', 'component']):
                return 'silkscreen_top'  # Default to top if ambiguous
            elif any(term in content_lower for term in ['paste', 'stencil']):
                return 'paste_top'  # Default to top if ambiguous
        
        # Check for Excellon drill format
        if any(indicator in content_lower for indicator in ['m48', 't1c', 'inch', 'metric', 'x', 'y']):
            return 'drill'
        
        return 'unknown'
    
    def _get_layer_description(self, layer_type: str) -> str:
        """Get human-readable description for layer type"""
        descriptions = {
            'copper_top': 'Top Copper',
            'copper_bottom': 'Bottom Copper',
            'copper_inner': 'Inner Copper',
            'soldermask_top': 'Top Soldermask',
            'soldermask_bottom': 'Bottom Soldermask',
            'silkscreen_top': 'Top Silkscreen',
            'silkscreen_bottom': 'Bottom Silkscreen',
            'paste_top': 'Top Paste',
            'paste_bottom': 'Bottom Paste',
            'outline': 'Board Outline',
            'drill': 'Drill File',
            'document': 'Documentation',
            'unknown': 'Unknown File Type'
        }
        return descriptions.get(layer_type, 'Unknown File Type')
    
    def get_layer_color(self, layer_type: str) -> str:
        """Get appropriate color for layer visualization"""
        colors = {
            'copper_top': '#CC6600',
            'copper_bottom': '#3366CC',
            'copper_inner': '#009900',
            'soldermask_top': '#006600',
            'soldermask_bottom': '#006600',
            'silkscreen_top': '#FFFFFF',
            'silkscreen_bottom': '#FFFFFF',
            'paste_top': '#C0C0C0',
            'paste_bottom': '#C0C0C0',
            'outline': '#000000',
            'drill': '#333333',
            'document': '#666666',
            'unknown': '#999999'
        }
        return colors.get(layer_type, '#999999')
    
    def analyze_file_set(self, filenames: List[str]) -> Dict[str, List[str]]:
        """
        Analyze a set of files and group them by layer type
        Useful for detecting complete PCB file sets
        """
        layer_groups = {}
        
        for filename in filenames:
            layer_type, description = self.detect_layer_type(filename)
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append(filename)
        
        return layer_groups
    
    def suggest_missing_layers(self, detected_layers: Dict[str, List[str]]) -> List[str]:
        """Suggest what layers might be missing from a PCB file set"""
        essential_layers = ['copper_top', 'copper_bottom', 'outline', 'drill']
        common_layers = ['soldermask_top', 'soldermask_bottom', 'silkscreen_top', 'silkscreen_bottom']
        
        missing = []
        
        # Check essential layers
        for layer in essential_layers:
            if layer not in detected_layers or not detected_layers[layer]:
                missing.append(f"Essential: {self._get_layer_description(layer)}")
        
        # Check common layers
        for layer in common_layers:
            if layer not in detected_layers or not detected_layers[layer]:
                missing.append(f"Common: {self._get_layer_description(layer)}")
        
        return missing


'''
class GerberParser:
    """Simple Gerber file parser"""
    
    def __init__(self, layer_detector: EnhancedLayerDetector):
        self.apertures = {}
        self.current_aperture = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.geometries = []
        self.units = 'mm'
        self.format_spec = (3, 3)  # Default format
        self.layer_detector = layer_detector
        
    def parse_file(self, filepath: str) -> PCBLayer:
        """Parse a Gerber file and return a PCBLayer object"""
        self.apertures = {}
        self.geometries = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            self._parse_line(line)
        
        # Calculate bounds
        bounds = self._calculate_bounds()
        
        # Use enhanced layer detection
        layer_type, description = self.layer_detector.detect_layer_type(filepath, content)
        
        return PCBLayer(
            name=os.path.basename(filepath),
            type=layer_type,
            filename=filepath,
            apertures=self.apertures,
            geometries=self.geometries,
            bounds=bounds,
            description=description
        )
    
    def _parse_line(self, line: str):
        """Parse a single line of Gerber code"""
        # Aperture definition
        if line.startswith('%ADD'):
            self._parse_aperture_definition(line)
        
        # Format specification
        elif line.startswith('%FSLAX'):
            self._parse_format_spec(line)
        
        # Units
        elif line.startswith('%MOMM'):
            self.units = 'mm'
        elif line.startswith('%MOIN'):
            self.units = 'inch'
        
        # Aperture selection
        elif line.startswith('D') and line[1:].isdigit():
            self.current_aperture = line
        
        # Coordinate and operation
        elif 'X' in line or 'Y' in line:
            self._parse_coordinate_line(line)
    
    def _parse_aperture_definition(self, line: str):
        """Parse aperture definition like %ADD10C,0.152400*%"""
        match = re.match(r'%ADD(\d+)([CR]),([\d.]+)(?:X([\d.]+))?', line)
        if match:
            aperture_id = f"D{match.group(1)}"
            shape = match.group(2)
            size1 = float(match.group(3))
            size2 = float(match.group(4)) if match.group(4) else size1
            
            self.apertures[aperture_id] = {
                'shape': 'circle' if shape == 'C' else 'rectangle',
                'size1': size1,
                'size2': size2
            }
    
    def _parse_format_spec(self, line: str):
        """Parse format specification like %FSLAX23Y23*%"""
        match = re.search(r'X(\d)(\d)Y(\d)(\d)', line)
        if match:
            self.format_spec = (int(match.group(1)), int(match.group(2)))
    
    def _parse_coordinate_line(self, line: str):
        """Parse coordinate and drawing commands"""
        # Extract coordinates
        x_match = re.search(r'X([-]?\d+)', line)
        y_match = re.search(r'Y([-]?\d+)', line)
        
        if x_match:
            x_val = int(x_match.group(1))
            self.current_x = x_val / (10 ** sum(self.format_spec))
        
        if y_match:
            y_val = int(y_match.group(1))
            self.current_y = y_val / (10 ** sum(self.format_spec))
        
        # Check for operations
        if line.endswith('D01*'):  # Interpolate (draw)
            self._add_geometry('line', self.current_x, self.current_y)
        elif line.endswith('D02*'):  # Move
            pass  # Just update position
        elif line.endswith('D03*'):  # Flash
            self._add_geometry('flash', self.current_x, self.current_y)
    
    def _add_geometry(self, operation: str, x: float, y: float):
        """Add geometry to the list"""
        if self.current_aperture and self.current_aperture in self.apertures:
            self.geometries.append({
                'operation': operation,
                'x': x,
                'y': y,
                'aperture': self.current_aperture
            })
    
    def _calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate the bounding box of all geometries"""
        if not self.geometries:
            return (0, 0, 0, 0)
        
        x_coords = [g['x'] for g in self.geometries]
        y_coords = [g['y'] for g in self.geometries]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
'''

class DrillParser:
    """Simple Excellon drill file parser"""
    
    def __init__(self):
        self.tools = {}
        self.holes = []
        self.units = 'inch'  # Default for Excellon
        
    def parse_file(self, filepath: str) -> List[DrillHole]:
        """Parse an Excellon drill file"""
        self.tools = {}
        self.holes = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        
        lines = content.split('\n')
        current_tool = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Tool definition
            if line.startswith('T') and 'C' in line:
                self._parse_tool_definition(line)
            
            # Tool selection
            elif line.startswith('T') and line[1:].isdigit():
                current_tool = line
            
            # Units
            elif line in ['INCH', 'METRIC']:
                self.units = 'inch' if line == 'INCH' else 'mm'
            
            # Coordinates
            elif line.startswith('X') or line.startswith('Y'):
                if current_tool:
                    self._parse_coordinate(line, current_tool)
        
        return self.holes
    
    def _parse_tool_definition(self, line: str):
        """Parse tool definition like T01C0.0135"""
        match = re.match(r'T(\d+)C([\d.]+)', line)
        if match:
            tool_id = f"T{match.group(1).zfill(2)}"
            diameter = float(match.group(2))
            self.tools[tool_id] = diameter
    
    def _parse_coordinate(self, line: str, tool: str):
        """Parse coordinate line and add hole"""
        x_match = re.search(r'X([-]?[\d.]+)', line)
        y_match = re.search(r'Y([-]?[\d.]+)', line)
        
        if x_match and y_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            diameter = self.tools.get(tool, 0.1)  # Default diameter
            
            # Convert to mm if needed
            if self.units == 'inch':
                x *= 25.4
                y *= 25.4
                diameter *= 25.4
            
            self.holes.append(DrillHole(x, y, diameter))


'''
class LayerVisualizer:
    """Create visualizations of PCB layers"""
    
    def __init__(self, layer_detector: EnhancedLayerDetector):
        self.layer_detector = layer_detector
    
    def render_layer(self, layer: PCBLayer, width: int = 800, height: int = 600) -> str:
        """Render a PCB layer and return base64-encoded PNG"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        color = self.layer_detector.get_layer_color(layer.type)
        
        # Render geometries
        for geom in layer.geometries:
            self._render_geometry(ax, geom, layer.apertures, color)
        
        # Set bounds with some padding
        if layer.bounds != (0, 0, 0, 0):
            min_x, min_y, max_x, max_y = layer.bounds
            padding = max((max_x - min_x), (max_y - min_y)) * 0.1
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
        
        ax.set_aspect('equal')
        ax.set_facecolor('black' if layer.type in ['copper_top', 'copper_bottom'] else 'darkgreen')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
        
    def render_drill_layer(self, holes: List[DrillHole], 
                          bounds: Tuple[float, float, float, float],
                          width: int = 800, height: int = 600) -> str:
        """Render drill holes and return base64-encoded PNG"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        for hole in holes:
            circle = Circle((hole.x, hole.y), hole.diameter/2, 
                          color='white', fill=True, alpha=0.8)
            ax.add_patch(circle)
            
            # Add hole outline
            circle_outline = Circle((hole.x, hole.y), hole.diameter/2, 
                                  color='black', fill=False, linewidth=0.5)
            ax.add_patch(circle_outline)
        
        # Set bounds
        if bounds != (0, 0, 0, 0):
            min_x, min_y, max_x, max_y = bounds
            padding = max((max_x - min_x), (max_y - min_y)) * 0.1
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
        
        ax.set_aspect('equal')
        ax.set_facecolor('darkgreen')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor='darkgreen', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    def _render_geometry(self, ax, geometry: Dict[str, Any], 
                        apertures: Dict[str, Any], color: str):
        """Render a single geometry element"""
        aperture_id = geometry.get('aperture')
        if not aperture_id or aperture_id not in apertures:
            return
        
        aperture = apertures[aperture_id]
        x, y = geometry['x'], geometry['y']
        
        if geometry['operation'] == 'flash':
            if aperture['shape'] == 'circle':
                circle = Circle((x, y), aperture['size1']/2, 
                              color=color, alpha=0.8)
                ax.add_patch(circle)
            elif aperture['shape'] == 'rectangle':
                rect = Rectangle((x - aperture['size1']/2, y - aperture['size2']/2),
                               aperture['size1'], aperture['size2'],
                               color=color, alpha=0.8)
                ax.add_patch(rect)
'''


class EnhancedGerberParser:
    """Enhanced Gerber file parser with better geometry extraction"""
    
    def __init__(self, layer_detector: EnhancedLayerDetector):
        self.apertures = {}
        self.current_aperture = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.previous_x = 0.0
        self.previous_y = 0.0
        self.geometries = []
        self.units = 'mm'
        self.format_spec = (3, 3)  # Default format
        self.layer_detector = layer_detector
        self.interpolation_mode = 'linear'
        self.coordinate_format = 'absolute'
        self.zero_omission = 'leading'
    
    def parse_file(self, filepath: str) -> PCBLayer:
        """Parse a Gerber file and return a PCBLayer object"""
        self.apertures = {}
        self.geometries = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.previous_x = 0.0
        self.previous_y = 0.0
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                if line.startswith('%'):
                    self._parse_extended_command(line)
                continue
                
            self._parse_line(line)
        
        # Calculate bounds
        bounds = self._calculate_bounds()
        
        # Use enhanced layer detection
        layer_type, description = self.layer_detector.detect_layer_type(filepath, content)
        
        return PCBLayer(
            name=os.path.basename(filepath),
            type=layer_type,
            filename=filepath,
            apertures=self.apertures,
            geometries=self.geometries,
            bounds=bounds,
            description=description
        )
    
    def _parse_extended_command(self, line: str):
        """Parse extended commands (those starting with %)"""
        # Aperture definition
        if line.startswith('%ADD'):
            self._parse_aperture_definition(line)
        
        # Format specification
        elif line.startswith('%FSLAX') or line.startswith('%FSLAY'):
            self._parse_format_spec(line)
        
        # Units
        elif line.startswith('%MOMM'):
            self.units = 'mm'
        elif line.startswith('%MOIN'):
            self.units = 'inch'
        
        # Load polarity
        elif line.startswith('%LPD'):
            pass  # Dark polarity (default)
        elif line.startswith('%LPC'):
            pass  # Clear polarity
    
    def _parse_line(self, line: str):
        """Parse a single line of Gerber code"""
        # Skip empty lines and comments
        if not line or line.startswith('G04'):
            return
        
        # Aperture selection
        if re.match(r'^D\d+\*?$', line):
            aperture_code = line.rstrip('*')
            if aperture_code in self.apertures:
                self.current_aperture = aperture_code
            return
        
        # Interpolation mode
        if line.startswith('G01'):
            self.interpolation_mode = 'linear'
        elif line.startswith('G02'):
            self.interpolation_mode = 'clockwise_arc'
        elif line.startswith('G03'):
            self.interpolation_mode = 'counterclockwise_arc'
        
        # Coordinate and operation
        if 'X' in line or 'Y' in line or line.endswith('D01*') or line.endswith('D02*') or line.endswith('D03*'):
            self._parse_coordinate_line(line)
    
    def _parse_aperture_definition(self, line: str):
        """Parse aperture definition like %ADD10C,0.152400*%"""
        # Remove % and * characters
        line = line.strip('%*')
        
        # Match different aperture types
        circle_match = re.match(r'ADD(\d+)C,([\d.]+)(?:X([\d.]+))?', line)
        rect_match = re.match(r'ADD(\d+)R,([\d.]+)X([\d.]+)(?:X([\d.]+))?', line)
        obround_match = re.match(r'ADD(\d+)O,([\d.]+)X([\d.]+)(?:X([\d.]+))?', line)
        polygon_match = re.match(r'ADD(\d+)P,([\d.]+)X(\d+)(?:X([\d.]+))?', line)
        
        if circle_match:
            aperture_id = f"D{circle_match.group(1)}"
            diameter = float(circle_match.group(2))
            hole_diameter = float(circle_match.group(3)) if circle_match.group(3) else 0
            
            self.apertures[aperture_id] = {
                'shape': 'circle',
                'size1': diameter,
                'size2': diameter,
                'hole_diameter': hole_diameter
            }
        
        elif rect_match:
            aperture_id = f"D{rect_match.group(1)}"
            width = float(rect_match.group(2))
            height = float(rect_match.group(3))
            hole_diameter = float(rect_match.group(4)) if rect_match.group(4) else 0
            
            self.apertures[aperture_id] = {
                'shape': 'rectangle',
                'size1': width,
                'size2': height,
                'hole_diameter': hole_diameter
            }
        
        elif obround_match:
            aperture_id = f"D{obround_match.group(1)}"
            width = float(obround_match.group(2))
            height = float(obround_match.group(3))
            hole_diameter = float(obround_match.group(4)) if obround_match.group(4) else 0
            
            self.apertures[aperture_id] = {
                'shape': 'obround',
                'size1': width,
                'size2': height,
                'hole_diameter': hole_diameter
            }
        
        elif polygon_match:
            aperture_id = f"D{polygon_match.group(1)}"
            diameter = float(polygon_match.group(2))
            vertices = int(polygon_match.group(3))
            rotation = float(polygon_match.group(4)) if polygon_match.group(4) else 0
            
            self.apertures[aperture_id] = {
                'shape': 'polygon',
                'size1': diameter,
                'size2': diameter,
                'vertices': vertices,
                'rotation': rotation
            }
    
    def _parse_format_spec(self, line: str):
        """Parse format specification like %FSLAX23Y23*%"""
        # Extract format information
        if 'X' in line and 'Y' in line:
            x_match = re.search(r'X(\d)(\d)', line)
            if x_match:
                integer_places = int(x_match.group(1))
                decimal_places = int(x_match.group(2))
                self.format_spec = (integer_places, decimal_places)
        
        # Determine zero omission
        if 'L' in line:
            self.zero_omission = 'leading'
        elif 'T' in line:
            self.zero_omission = 'trailing'
        
        # Determine coordinate format
        if 'A' in line:
            self.coordinate_format = 'absolute'
        elif 'I' in line:
            self.coordinate_format = 'incremental'
    
    def _parse_coordinate_line(self, line: str):
        """Parse coordinate and drawing commands"""
        # Store previous position
        self.previous_x = self.current_x
        self.previous_y = self.current_y
        
        # Extract coordinates
        x_match = re.search(r'X([-+]?\d+)', line)
        y_match = re.search(r'Y([-+]?\d+)', line)
        
        if x_match:
            x_val = int(x_match.group(1))
            self.current_x = self._convert_coordinate(x_val)
        
        if y_match:
            y_val = int(y_match.group(1))
            self.current_y = self._convert_coordinate(y_val)
        
        # Handle operations
        if line.endswith('D01*'):  # Interpolate (draw)
            self._add_line_geometry()
        elif line.endswith('D02*'):  # Move
            pass  # Just update position, no drawing
        elif line.endswith('D03*'):  # Flash
            self._add_flash_geometry()
    
    def _convert_coordinate(self, raw_value: int) -> float:
        """Convert raw coordinate value to actual coordinate"""
        # Apply format specification
        divisor = 10 ** self.format_spec[1]
        coordinate = raw_value / divisor
        
        # Convert units if necessary
        if self.units == 'inch':
            coordinate *= 25.4  # Convert to mm
        
        return coordinate
    
    def _add_line_geometry(self):
        """Add line geometry from previous position to current position"""
        if self.current_aperture and self.current_aperture in self.apertures:
            # Only add if we have moved
            if (abs(self.current_x - self.previous_x) > 0.001 or 
                abs(self.current_y - self.previous_y) > 0.001):
                
                self.geometries.append({
                    'operation': 'line',
                    'x1': self.previous_x,
                    'y1': self.previous_y,
                    'x2': self.current_x,
                    'y2': self.current_y,
                    'aperture': self.current_aperture
                })
    
    def _add_flash_geometry(self):
        """Add flash geometry at current position"""
        if self.current_aperture and self.current_aperture in self.apertures:
            self.geometries.append({
                'operation': 'flash',
                'x': self.current_x,
                'y': self.current_y,
                'aperture': self.current_aperture
            })
    
    def _calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate the bounding box of all geometries"""
        if not self.geometries:
            return (0, 0, 0, 0)
        
        x_coords = []
        y_coords = []
        
        for geom in self.geometries:
            if geom['operation'] == 'flash':
                x_coords.append(geom['x'])
                y_coords.append(geom['y'])
            elif geom['operation'] == 'line':
                x_coords.extend([geom['x1'], geom['x2']])
                y_coords.extend([geom['y1'], geom['y2']])
        
        if not x_coords:
            return (0, 0, 0, 0)
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

class EnhancedLayerVisualizer:
    """Enhanced visualizer for better PCB layer rendering"""
    
    def __init__(self, layer_detector: EnhancedLayerDetector):
        self.layer_detector = layer_detector
    
    def render_layer(self, layer: PCBLayer, width: int = 800, height: int = 600) -> str:
        """Render a PCB layer with enhanced visualization"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        color = self.layer_detector.get_layer_color(layer.type)
        background_color = self._get_background_color(layer.type)
        
        # Render geometries
        for geom in layer.geometries:
            self._render_enhanced_geometry(ax, geom, layer.apertures, color)
        
        # Set bounds with padding
        if layer.bounds != (0, 0, 0, 0):
            min_x, min_y, max_x, max_y = layer.bounds
            # Add 10% padding
            width_padding = (max_x - min_x) * 0.1
            height_padding = (max_y - min_y) * 0.1
            ax.set_xlim(min_x - width_padding, max_x + width_padding)
            ax.set_ylim(min_y - height_padding, max_y + height_padding)
        else:
            # Default bounds if no geometry
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
        
        ax.set_aspect('equal')
        ax.set_facecolor(background_color)
        ax.axis('off')
        
        # Add title
        plt.title(f"{layer.description} ({len(layer.geometries)} features)", 
                 color='white', fontsize=10, pad=20)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor=background_color, edgecolor='none', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    def _get_background_color(self, layer_type: str) -> str:
        """Get appropriate background color for layer type"""
        backgrounds = {
            'copper_top': '#1a1a1a',      # Dark for copper
            'copper_bottom': '#1a1a1a',   # Dark for copper
            'copper_inner': '#1a1a1a',    # Dark for copper
            'soldermask_top': '#0d4d0d',  # Dark green for soldermask
            'soldermask_bottom': '#0d4d0d',
            'silkscreen_top': '#1a4d1a',  # Medium green for silkscreen
            'silkscreen_bottom': '#1a4d1a',
            'paste_top': '#2d2d2d',       # Dark gray for paste
            'paste_bottom': '#2d2d2d',
            'outline': '#ffffff',         # White for outline
            'drill': '#1a4d1a',          # Green for drill
        }
        return backgrounds.get(layer_type, '#1a1a1a')
    
    def _render_enhanced_geometry(self, ax, geometry: Dict[str, Any], 
                                apertures: Dict[str, Any], color: str):
        """Render geometry with enhanced visualization"""
        aperture_id = geometry.get('aperture')
        if not aperture_id or aperture_id not in apertures:
            return
        
        aperture = apertures[aperture_id]
        
        if geometry['operation'] == 'flash':
            self._render_flash(ax, geometry, aperture, color)
        elif geometry['operation'] == 'line':
            self._render_line(ax, geometry, aperture, color)
    
    def _render_flash(self, ax, geometry: Dict[str, Any], 
                     aperture: Dict[str, Any], color: str):
        """Render a flash (pad) with proper shape"""
        x, y = geometry['x'], geometry['y']
        
        if aperture['shape'] == 'circle':
            radius = aperture['size1'] / 2
            circle = Circle((x, y), radius, color=color, alpha=0.8, linewidth=0)
            ax.add_patch(circle)
            
            # Add hole if present
            if aperture.get('hole_diameter', 0) > 0:
                hole_radius = aperture['hole_diameter'] / 2
                hole = Circle((x, y), hole_radius, color='black', alpha=1.0)
                ax.add_patch(hole)
        
        elif aperture['shape'] == 'rectangle':
            width = aperture['size1']
            height = aperture['size2']
            rect = Rectangle((x - width/2, y - height/2), width, height,
                           color=color, alpha=0.8, linewidth=0)
            ax.add_patch(rect)
            
            # Add hole if present
            if aperture.get('hole_diameter', 0) > 0:
                hole_radius = aperture['hole_diameter'] / 2
                hole = Circle((x, y), hole_radius, color='black', alpha=1.0)
                ax.add_patch(hole)
        
        elif aperture['shape'] == 'obround':
            width = aperture['size1']
            height = aperture['size2']
            
            if width > height:
                # Horizontal obround
                rect_width = width - height
                rect = Rectangle((x - rect_width/2, y - height/2), rect_width, height,
                               color=color, alpha=0.8, linewidth=0)
                ax.add_patch(rect)
                
                # Add semicircles
                left_circle = Circle((x - rect_width/2, y), height/2, 
                                   color=color, alpha=0.8, linewidth=0)
                right_circle = Circle((x + rect_width/2, y), height/2, 
                                    color=color, alpha=0.8, linewidth=0)
                ax.add_patch(left_circle)
                ax.add_patch(right_circle)
            else:
                # Vertical obround
                rect_height = height - width
                rect = Rectangle((x - width/2, y - rect_height/2), width, rect_height,
                               color=color, alpha=0.8, linewidth=0)
                ax.add_patch(rect)
                
                # Add semicircles
                bottom_circle = Circle((x, y - rect_height/2), width/2, 
                                     color=color, alpha=0.8, linewidth=0)
                top_circle = Circle((x, y + rect_height/2), width/2, 
                                  color=color, alpha=0.8, linewidth=0)
                ax.add_patch(bottom_circle)
                ax.add_patch(top_circle)
    
    def _render_line(self, ax, geometry: Dict[str, Any], 
                    aperture: Dict[str, Any], color: str):
        """Render a line (trace) with proper width"""
        x1, y1 = geometry['x1'], geometry['y1']
        x2, y2 = geometry['x2'], geometry['y2']
        
        # Get line width from aperture
        if aperture['shape'] == 'circle':
            linewidth = aperture['size1']
        else:
            linewidth = min(aperture['size1'], aperture['size2'])
        
        # Convert mm to points (approximate)
        linewidth_points = linewidth * 2.83465  # 1mm ≈ 2.83 points
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth_points, 
                solid_capstyle='round', alpha=0.8)
        

        
class PCBFileManager:
    """Manage PCB file operations"""
    
    def __init__(self):
        self.layer_detector = EnhancedLayerDetector()
        self.gerber_parser = EnhancedGerberParser(self.layer_detector)
        self.drill_parser = DrillParser()
        self.visualizer = EnhancedLayerVisualizer(self.layer_detector)
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def process_uploaded_files(self, files: List[FileStorage], upload_dir: str) -> Dict[str, Any]:
        """Process uploaded PCB files"""
        results = {
            'layers': [],
            'drill_files': [],
            'other_files': [],
            'errors': []
        }
        
        for file in files:
            if file and file.filename and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_dir, filename)
                
                try:
                    file.save(filepath)
                    
                    if filename.lower().endswith('.zip'):
                        # Extract and process ZIP file
                        extracted_results = self._process_zip_file(filepath, upload_dir)
                        results['layers'].extend(extracted_results['layers'])
                        results['drill_files'].extend(extracted_results['drill_files'])
                        results['other_files'].extend(extracted_results['other_files'])
                        results['errors'].extend(extracted_results['errors'])
                    else:
                        # Process individual file
                        file_result = self._process_single_file(filepath)
                        if file_result['type'] == 'layer':
                            results['layers'].append(file_result)
                        elif file_result['type'] == 'drill':
                            results['drill_files'].append(file_result)
                        else:
                            results['other_files'].append(file_result)
                
                except Exception as e:
                    results['errors'].append(f"Error processing {filename}: {str(e)}")
        
        return results
    
    def _process_zip_file(self, zip_path: str, extract_dir: str) -> Dict[str, Any]:
        """Extract and process ZIP file contents"""
        results = {
            'layers': [],
            'drill_files': [],
            'other_files': [],
            'errors': []
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
                for filename in zip_ref.namelist():
                    if self.allowed_file(filename):
                        filepath = os.path.join(extract_dir, filename)
                        if os.path.isfile(filepath):
                            try:
                                file_result = self._process_single_file(filepath)
                                if file_result['type'] == 'layer':
                                    results['layers'].append(file_result)
                                elif file_result['type'] == 'drill':
                                    results['drill_files'].append(file_result)
                                else:
                                    results['other_files'].append(file_result)
                            except Exception as e:
                                results['errors'].append(f"Error processing {filename}: {str(e)}")
        
        except Exception as e:
            results['errors'].append(f"Error extracting ZIP file: {str(e)}")
        
        return results
    
    def _process_single_file(self, filepath: str) -> Dict[str, Any]:
        """Process a single PCB file"""
        filename = os.path.basename(filepath)
        file_ext = filename.lower().split('.')[-1]

        # Always try to parse as drill file first if extension matches
        if file_ext in ['drl', 'xln', 'txt', 'cnc', 'nc', 'tap']:
            try:
                holes = self.drill_parser.parse_file(filepath)
                if holes:
                    x_coords = [h.x for h in holes]
                    y_coords = [h.y for h in holes]
                    bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                    image_data = self.visualizer.render_drill_layer(holes, bounds)
                    return {
                        'type': 'drill',
                        'filename': filename,
                        'filepath': filepath,
                        'hole_count': len(holes),
                        'bounds': bounds,
                        'image_data': image_data
                    }
            except Exception as e:
                pass  # Fall through to other file type

            # --- Here is where you would call the advanced parser if needed ---
            # from drill_preview import parse_advanced_excellon, is_excellon_drill_file
            # if is_excellon_drill_file(filepath):
            #     holes = parse_advanced_excellon(filepath)
            #     if holes:
            #         # ...same as above, return drill file dict...

        # Use enhanced detector for other types
        layer_type, description = self.layer_detector.detect_layer_type(filepath)

        # Try to parse as Gerber file if it's detected as a PCB layer type
        if layer_type in [
            'copper_top', 'copper_bottom', 'copper_inner', 'soldermask_top',
            'soldermask_bottom', 'silkscreen_top', 'silkscreen_bottom',
            'paste_top', 'paste_bottom', 'outline'
        ] or file_ext in [
            'gbr', 'ger', 'gtl', 'gbl', 'gts', 'gbs', 'gto', 'gbo', 'gtp', 'gbp'
        ]:
            try:
                layer = self.gerber_parser.parse_file(filepath)
                image_data = None
                if layer.geometries:
                    image_data = self.visualizer.render_layer(layer)
                return {
                    'type': 'layer',
                    'filename': filename,
                    'filepath': filepath,
                    'layer_type': layer.type,
                    'geometry_count': len(layer.geometries),
                    'aperture_count': len(layer.apertures),
                    'bounds': layer.bounds,
                    'image_data': image_data
                }
            except Exception as e:
                if layer_type != 'unknown':
                    return {
                        'type': 'layer',
                        'filename': filename,
                        'filepath': filepath,
                        'layer_type': layer_type,
                        'geometry_count': 0,
                        'aperture_count': 0,
                        'bounds': (0, 0, 0, 0),
                        'image_data': None
                    }

        # Default to other file type
        return {
            'type': 'other',
            'filename': filename,
            'filepath': filepath,
            'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }


def create_enhanced_file_manager():
    """Create file manager with enhanced parsers"""
    class EnhancedPCBFileManager(PCBFileManager):
        def __init__(self):
            self.layer_detector = EnhancedLayerDetector()
            self.gerber_parser = EnhancedGerberParser(self.layer_detector)  # Use enhanced parser
            self.drill_parser = DrillParser()
            self.visualizer = EnhancedLayerVisualizer(self.layer_detector)  # Use enhanced visualizer
    
    return EnhancedPCBFileManager()

# Initialize file manager
file_manager = create_enhanced_file_manager()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload"""
    if 'files[]' not in request.files:
        flash('No files selected')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files[]')
    
    if not files or all(f.filename == '' for f in files):
        flash('No files selected')
        return redirect(url_for('index'))
    
    # Create temporary directory for this upload session
    upload_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
    
    try:
        results = file_manager.process_uploaded_files(files, upload_dir)
        
        if results['errors']:
            for error in results['errors']:
                flash(error, 'error')
        
        if not any([results['layers'], results['drill_files'], results['other_files']]):
            flash('No valid PCB files found in upload', 'error')
            return redirect(url_for('index'))
        
        # Store results in session or return directly
        return render_template('preview.html', results=results)
    
    except Exception as e:
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/file-info/<path:filename>')
def get_file_info(filename):
    """Get detailed information about a specific file"""
    # This would be implemented to return detailed file information
    return jsonify({'filename': filename, 'info': 'File info would go here'})

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download a processed file"""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)