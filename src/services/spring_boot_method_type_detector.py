import re
from typing import List, Tuple
from models.analyzer_config import AnalyzerConfig
from models.domain_models import MethodType

class SpringBootMethodTypeDetector:
    """Detects method types for Spring Boot applications."""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
    
    def detect_method_type(self, method_name: str, is_constructor: bool, body: str, 
                          annotations: List[str], return_type: str, parameters: List[Tuple[str, str]]) -> MethodType:
        """Determine method type with improved classification."""
        if is_constructor:
            return MethodType.CONSTRUCTOR
        
        rest_annotations = {'@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping', 
                          '@PatchMapping', '@RequestMapping'}
        if any(anno.split('(')[0] in rest_annotations for anno in annotations):
            return MethodType.REST_ENDPOINT
        
        if method_name.startswith('get') and len(method_name) > 3 and method_name[3].isupper():
            if len(parameters) == 0 and return_type != 'void':
                return MethodType.GETTER
        elif method_name.startswith('is') and len(method_name) > 2 and method_name[2].isupper():
            if len(parameters) == 0 and return_type in ['boolean', 'Boolean']:
                return MethodType.GETTER
        elif method_name.startswith('set') and len(method_name) > 3 and method_name[3].isupper():
            if len(parameters) == 1 and return_type == 'void':
                return MethodType.SETTER
        
        combined_text = " ".join(annotations) + " " + body
        
        if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in self.config.job_patterns):
            return MethodType.JOB_CONFIG
        
        if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in self.config.publisher_patterns):
            return MethodType.PUBLISHER_CONFIG
        
        if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in self.config.listener_patterns):
            return MethodType.LISTENER_CONFIG
        
        return MethodType.REGULAR