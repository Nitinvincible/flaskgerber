#!/usr/bin/env python3
"""
Enhanced PCB Layer Detection System
Supports comprehensive layer type identification for various PCB manufacturers and design tools
"""

import re
from typing import Dict, List, Tuple, Optional

class EnhancedLayerDetector:
    """Enhanced layer type detection supporting multiple naming conventions"""
    
    def __init__(self):
        # Comprehensive layer mapping patterns
        # Each pattern is (regex_pattern, layer_type, description)
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
            
            # Inner copper layers (map to top/bottom based on layer number)
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
            (r'.*[-_\.]?sp[-_\.]?(top|t|front|f).*', 'paste_top', 'Top Paste'),
            
            # Bottom paste  
            (r'.*\.(gbp|b\.paste|paste_bottom|bottompaste|pb)$', 'paste_bottom', 'Bottom Paste'),
            (r'.*[-_\.]?(bottom|back)[-_\.]?(paste|stencil).*', 'paste_bottom', 'Bottom Paste'),
            (r'.*[-_\.]?(paste|stencil)[-_\.]?(bottom|back).*', 'paste_bottom', 'Bottom Paste'),
            (r'.*[-_\.]?sp[-_\.]?(bottom|b|back).*', 'paste_bottom', 'Bottom Paste'),
            
            # === OUTLINE/MECHANICAL ===
            (r'.*\.(gko|gm1|out|mill|outline|edge|mechanical|board|cutout)$', 'outline', 'Board Outline'),
            (r'.*[-_\.]?(outline|edge|mill|milling|cutout|mechanical|board).*', 'outline', 'Board Outline'),
            (r'.*[-_\.]?edge[-_\.]?cuts.*', 'outline', 'Board Outline'),
            (r'.*[-_\.]?(keep|keepout)[-_\.]?out.*', 'outline', 'Board Outline'),
            
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
            
            # Altium common patterns
            'boardlayers.pdf': 'document',
            'drillguide.pdf': 'document',
            'assembly.pdf': 'document',
            'fab.pdf': 'document',
            
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
        
        # Priority order for layer types (higher index = higher priority)
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

# Example usage and testing
if __name__ == "__main__":
    detector = EnhancedLayerDetector()
    
    # Test various filename patterns
    test_files = [
        # EasyEDA exports
        "copper_top.gbr",
        "copper_bottom.gbr", 
        "soldermask_top.gbr",
        "soldermask_bottom.gbr",
        "silkscreen_top.gbr",
        "silkscreen_bottom.gbr",
        "pastestencil_top.gbr",
        "pastestencil_bottom.gbr",
        "boardoutline.gbr",
        "drill.drl",
        
        # KiCad exports
        "project-F_Cu.gbr",
        "project-B_Cu.gbr",
        "project-F_Mask.gbr", 
        "project-B_Mask.gbr",
        "project-F_SilkS.gbr",
        "project-B_SilkS.gbr",
        "project-F_Paste.gbr",
        "project-B_Paste.gbr",
        "project-Edge_Cuts.gbr",
        "project.drl",
        
        # Altium exports
        "PCB1.GTL",
        "PCB1.GBL",
        "PCB1.GTS",
        "PCB1.GBS", 
        "PCB1.GTO",
        "PCB1.GBO",
        "PCB1.GTP",
        "PCB1.GBP",
        "PCB1.GKO",
        "PCB1.DRL",
        
        # JLCPCB style
        "Gerber_TopLayer.GTL",
        "Gerber_BottomLayer.GBL",
        "Gerber_TopSolderMaskLayer.GTS",
        "Gerber_BottomSolderMaskLayer.GBS",
        "Gerber_TopSilkLayer.GTO",
        "Gerber_BottomSilkLayer.GBO",
        "Drill_PTH_Through.DRL",
        
        # Eagle exports
        "board.cmp",
        "board.sol",
        "board.stc",
        "board.sts",
        "board.plc", 
        "board.pls",
        "board.crc",
        "board.crs",
        "board.mil",
        "board.drd",
        
        # Various other patterns
        "top_copper.ger",
        "bottom_copper.ger",
        "top_mask.ger",
        "drill_file.xln",
        "outline.gbr",
        "assembly.pdf",
        "bom.csv",
        "placement.xy"
    ]
    
    print("Testing layer detection:")
    print("=" * 50)
    
    for filename in test_files:
        layer_type, description = detector.detect_layer_type(filename)
        color = detector.get_layer_color(layer_type)
        print(f"{filename:30} -> {layer_type:15} ({description}) [{color}]")
    
    print("\n" + "=" * 50)
    print("File set analysis:")
    
    layer_groups = detector.analyze_file_set(test_files)
    for layer_type, files in layer_groups.items():
        print(f"\n{layer_type.upper()}:")
        for file in files:
            print(f"  - {file}")
    
    print("\n" + "=" * 50)
    print("Missing layer suggestions:")
    missing = detector.suggest_missing_layers(layer_groups)
    for suggestion in missing:
        print(f"  - {suggestion}")