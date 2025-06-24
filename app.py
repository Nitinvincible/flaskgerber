#!/usr/bin/env python3
"""
Advanced PCB File Preview Flask Application
Supports Gerber (.gbr, .ger), Excellon (.drl, .xln), Pick & Place (.pnp), and other PCB formats
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
    'drl', 'xln', 'txt', 'cnc', 'nc', 'tap',  # Drill files
    'pnp', 'pos', 'xy',  # Pick and place
    'zip', 'rar',  # Archives
    'pcb', 'sch',  # PCB design files
    'pdf', 'svg'  # Documentation
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

@dataclass
class DrillHole:
    """Represents a drill hole"""
    x: float
    y: float
    diameter: float
    plated: bool = True

class GerberParser:
    """Simple Gerber file parser"""
    
    def __init__(self):
        self.apertures = {}
        self.current_aperture = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.geometries = []
        self.units = 'mm'
        self.format_spec = (3, 3)  # Default format
        
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
        layer_type = self._guess_layer_type(filepath)
        
        return PCBLayer(
            name=os.path.basename(filepath),
            type=layer_type,
            filename=filepath,
            apertures=self.apertures,
            geometries=self.geometries,
            bounds=bounds
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
    
    def _guess_layer_type(self, filepath: str) -> str:
        """Guess the layer type from filename"""
        filename = os.path.basename(filepath).lower()
        
        if any(ext in filename for ext in ['gtl', 'cmp', 'top']):
            return 'copper_top'
        elif any(ext in filename for ext in ['gbl', 'sol', 'bottom']):
            return 'copper_bottom'
        elif any(ext in filename for ext in ['gts', 'stc', 'tsc']):
            return 'soldermask_top'
        elif any(ext in filename for ext in ['gbs', 'sts', 'bsc']):
            return 'soldermask_bottom'
        elif any(ext in filename for ext in ['gto', 'plc', 'tsk']):
            return 'silkscreen_top'
        elif any(ext in filename for ext in ['gbo', 'pls', 'bsk']):
            return 'silkscreen_bottom'
        elif any(ext in filename for ext in ['gko', 'out', 'mill']):
            return 'outline'
        else:
            return 'unknown'

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

class LayerVisualizer:
    """Create visualizations of PCB layers"""
    
    def __init__(self):
        self.colors = {
            'copper_top': '#CC6600',
            'copper_bottom': '#3366CC',
            'soldermask_top': '#006600',
            'soldermask_bottom': '#006600',
            'silkscreen_top': '#FFFFFF',
            'silkscreen_bottom': '#FFFFFF',
            'outline': '#000000',
            'drill': '#333333',
            'unknown': '#999999'
        }
    
    def render_layer(self, layer: PCBLayer, width: int = 800, height: int = 600) -> str:
        """Render a PCB layer and return base64-encoded PNG"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        color = self.colors.get(layer.type, '#999999')
        
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

class PCBFileManager:
    """Manage PCB file operations"""
    
    def __init__(self):
        self.gerber_parser = GerberParser()
        self.drill_parser = DrillParser()
        self.visualizer = LayerVisualizer()
    
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
        
        if file_ext in ['drl', 'xln', 'txt', 'cnc', 'nc', 'tap']:
            # Try to parse as drill file
            try:
                holes = self.drill_parser.parse_file(filepath)
                if holes:
                    # Calculate bounds for drill file
                    x_coords = [h.x for h in holes]
                    y_coords = [h.y for h in holes]
                    bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                    
                    # Generate visualization
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
        
        # Try to parse as Gerber file
        try:
            layer = self.gerber_parser.parse_file(filepath)
            if layer.geometries:  # Only if we found some geometry
                # Generate visualization
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
            pass
        
        # Default to other file type
        return {
            'type': 'other',
            'filename': filename,
            'filepath': filepath,
            'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }

# Initialize file manager
file_manager = PCBFileManager()

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
    app.run(debug=True)
