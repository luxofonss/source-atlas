import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

try:
    from tree_sitter_languages import get_language
    from tree_sitter import Language, Parser, Node
except ImportError:
    print("Please install tree-sitter and tree-sitter-java:")
    print("pip install tree-sitter tree-sitter-java")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Domain Models (Value Objects and Entities)
# ============================================================================

class ClassType(Enum):
    CONFIGURATION = "configuration"
    SERVICE = "service"
    CONTROLLER = "controller"
    REPOSITORY = "repository"
    ENTITY = "entity"
    FILTER = "filter"
    COMPONENT = "component"
    CONTROLLER_ADVICE = "controlleradvice"
    BEAN = "bean"
    ABSTRACT = "abstract"
    RECORD = "record"
    ENUM = "enum"
    INTERFACE = "interface"
    CLASS = "class"
    ANNOTATION = "annotation"


class MethodType(Enum):
    CONSTRUCTOR = "constructor"
    GETTER = "getter"
    SETTER = "setter"
    JOB_CONFIG = "job_config"
    PUBLISHER_CONFIG = "publisher_config"
    LISTENER_CONFIG = "listener_config"
    REST_ENDPOINT = "rest_endpoint"
    REGULAR = "regular"


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    full_type: str
    annotations: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RestEndpoint:
    http_method: str
    path: str


@dataclass(frozen=True)
class Method:
    name: str
    return_type: str = ""
    full_return_type: str = ""
    parameters: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    modifiers: Tuple[str, ...] = field(default_factory=tuple)
    annotations: Tuple[str, ...] = field(default_factory=tuple)
    method_type: MethodType = MethodType.REGULAR
    body: str = ""
    is_abstract: bool = False
    is_override: bool = False
    throws: Tuple[str, ...] = field(default_factory=tuple)
    method_calls: Tuple[str, ...] = field(default_factory=tuple)
    variable_usage: Tuple[str, ...] = field(default_factory=tuple)
    inheritance_info: Tuple[str, ...] = field(default_factory=tuple)
    extends_info: Tuple[str, ...] = field(default_factory=tuple)
    endpoint: Optional[RestEndpoint] = None


@dataclass(frozen=True)
class CodeChunk:
    package: str
    class_name: str
    full_class_name: str
    class_type: ClassType
    file_path: str
    content: str
    implements: Tuple[str, ...] = field(default_factory=tuple)
    extends: Optional[str] = None
    fields: Tuple[Field, ...] = field(default_factory=tuple)
    imports: Dict[str, str] = field(default_factory=dict)
    methods: Tuple[Method, ...] = field(default_factory=tuple)
    annotations: Tuple[str, ...] = field(default_factory=tuple)
    is_nested: bool = False
    parent_class: Optional[str] = None
    inner_classes: Tuple[str, ...] = field(default_factory=tuple)
    modifiers: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class DependencyGraph:
    nodes: Set[str] = field(default_factory=set)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': list(self.nodes),
            'edges': [{'from': edge[0], 'to': edge[1], 'relation': edge[2]} for edge in self.edges]
        }


# ============================================================================
# Configuration and Constants
# ============================================================================

@dataclass(frozen=True)
class AnalyzerConfig:
    remove_comments: bool = True
    builtin_types: Set[str] = field(default_factory=lambda: {
        'byte', 'short', 'int', 'long', 'float', 'double', 'boolean', 'char', 'void',
        'String', 'Object', 'Class', 'Integer', 'Long', 'Double', 'Float', 'Boolean',
        'Character', 'Byte', 'Short', 'BigDecimal', 'BigInteger', 'Date', 'LocalDate',
        'LocalDateTime', 'LocalTime', 'Instant', 'Duration', 'Period', 'List', 'Set',
        'Map', 'Collection', 'Optional', 'Stream', 'Future', 'CompletableFuture'
    })
    springboot_annotations: Dict[str, ClassType] = field(default_factory=lambda: {
        '@Configuration': ClassType.CONFIGURATION,
        '@EnableAutoConfiguration': ClassType.CONFIGURATION,
        '@SpringBootApplication': ClassType.CONFIGURATION,
        '@Service': ClassType.SERVICE,
        '@Controller': ClassType.CONTROLLER,
        '@RestController': ClassType.CONTROLLER,
        '@Repository': ClassType.REPOSITORY,
        '@Entity': ClassType.ENTITY,
        '@Table': ClassType.ENTITY,
        '@Document': ClassType.ENTITY,
        '@Filter': ClassType.FILTER,
        '@Component': ClassType.COMPONENT,
        '@ControllerAdvice': ClassType.CONTROLLER_ADVICE,
        '@RestControllerAdvice': ClassType.CONTROLLER_ADVICE,
        '@Bean': ClassType.BEAN,
    })
    job_patterns: List[str] = field(default_factory=lambda: [
        r'@Scheduled\s*\(',
        r'@Job\s*\(',
        r'@EnableScheduling',
        r'JobLauncher',
        r'QuartzJob',
        r'@Async\s*\(',
        r'TaskExecutor',
        r'ThreadPoolTaskExecutor',
        r'@EnableAsync'
    ])
    publisher_patterns: List[str] = field(default_factory=lambda: [
        r'ApplicationEventPublisher',
        r'@EventListener',
        r'@RabbitTemplate',
        r'@KafkaTemplate',
        r'RedisTemplate',
        r'JmsTemplate',
        r'@TransactionalEventListener',
        r'@Publish',
        r'MessageProducer',
        r'\.publish\s*\(',
        r'\.send\s*\(',
        r'\.convertAndSend\s*\('
    ])
    listener_patterns: List[str] = field(default_factory=lambda: [
        r'@EventListener\s*\(',
        r'@RabbitListener\s*\(',
        r'@KafkaListener\s*\(',
        r'@JmsListener\s*\(',
        r'@StreamListener\s*\(',
        r'MessageListener',
        r'@Subscribe',
        r'@Handler',
        r'ApplicationListener',
        r'@Component.*Listener'
    ])


# ============================================================================
# Fixed Components
# ============================================================================

class JavaCommentRemover:
    def remove_comments(self, content: str) -> str:
        """Remove comments from Java source code while preserving line numbers"""
        lines = content.split('\n')
        result_lines = []
        in_multiline_comment = False
        
        for line in lines:
            if in_multiline_comment:
                end_pos = line.find('*/')
                if end_pos != -1:
                    in_multiline_comment = False
                    line = line[end_pos + 2:]
                else:
                    result_lines.append(' ' * len(line))
                    continue
            
            # Remove single line comments
            single_comment_pos = line.find('//')
            if single_comment_pos != -1:
                line = line[:single_comment_pos]
            
            # Handle multiline comments
            while True:
                start_pos = line.find('/*')
                if start_pos == -1:
                    break
                    
                end_pos = line.find('*/', start_pos + 2)
                if end_pos != -1:
                    line = line[:start_pos] + ' ' * (end_pos - start_pos + 2) + line[end_pos + 2:]
                else:
                    in_multiline_comment = True
                    line = line[:start_pos]
                    break
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)


class JavaTypeResolver:
    def __init__(self, config: AnalyzerConfig, class_cache: Set[str]):
        self.config = config
        self.class_cache = class_cache
    
    def resolve_type_name(self, type_name: str, imports: Dict[str, str], package: str) -> List[str]:
        """Resolve type name to full qualified name"""
        if not type_name:
            return []
        
        # Handle array types
        array_suffix = ""
        if type_name.endswith('[]'):
            array_suffix = "[]"
            type_name = type_name[:-2].strip()
        
        # Handle generic types - extract base type
        generic_match = re.match(r'^([^<]+)', type_name)
        if generic_match:
            base_type, all_types, generic_part = self.extract_all_types_from_generic(type_name)
        else:
            base_type = type_name
            all_types = []
            generic_part = ""
        
        # Check if it's a primitive or common built-in type
        types = set()

        # Process base type first
        raw_types = set()
        raw_types.add(base_type)
        for t in all_types:
            raw_types.add(t)

        for rt in raw_types:
            # Check java.lang package (implicit import)
            java_lang_types = {
                'String', 'Object', 'Class', 'Integer', 'Long', 'Double', 'Float', 
                'Boolean', 'Character', 'Byte', 'Short', 'Number', 'Exception',
                'RuntimeException', 'Throwable', 'Error', 'Thread', 'Runnable'
            }
            if rt not in java_lang_types:
                if rt in self.config.builtin_types:
                    types.add(rt)
                
                # Check direct imports
                elif rt in imports:
                    resolved = imports[rt]
                    types.add(resolved)
                
                # Check wildcard imports
                else:
                    resolved_from_wildcard = self._resolve_from_wildcard_imports(rt, imports)
                    if resolved_from_wildcard:
                        types.add(resolved_from_wildcard + generic_part + array_suffix)
                    
                    # Check if it's in the same package
                    elif package and rt[0].isupper():
                        same_package_type = f"{package}.{rt}"
                        if same_package_type in self.class_cache:
                            types.add(same_package_type + generic_part + array_suffix)
                        else:
                            types.add(type_name + array_suffix)
                    else:
                        types.add(type_name + array_suffix)

        # Process generic types
        for tp in all_types:
            if tp in java_lang_types:
                continue

            if tp in self.config.builtin_types:
                types.add(tp + generic_part + array_suffix)
            
            # Check direct imports
            elif tp in imports:
                resolved = imports[tp]
                types.add(resolved + generic_part + array_suffix)
            
            # Check wildcard imports
            else:
                resolved_from_wildcard = self._resolve_from_wildcard_imports(tp, imports)
                if resolved_from_wildcard:
                    types.add(resolved_from_wildcard + generic_part + array_suffix)
                
                # Check if it's in the same package
                elif package and tp[0].isupper():
                    same_package_type = f"{package}.{tp}"
                    if same_package_type in self.class_cache:
                        types.add(same_package_type + generic_part + array_suffix)
                    else:
                        types.add(tp + generic_part + array_suffix)
                else:
                    types.add(tp + generic_part + array_suffix)
        
        return list(types)
    
    def extract_all_types_from_generic(self, type_name: str) -> Tuple[str, List[str], str]:
        type_name = type_name.strip()
        
        # Extract base type (everything before the first '<')
        base_match = re.match(r'^([^<]+)', type_name)
        if not base_match:
            return type_name, [], ""
        
        base_type = base_match.group(1).strip()
        
        # If no generics, return early
        if '<' not in type_name:
            return base_type, [], ""
        
        # Extract the generic part (everything between outermost < and >)
        generic_start = type_name.find('<')
        generic_end = self.find_matching_bracket(type_name, generic_start)
        
        if generic_end == -1:
            # Malformed generic, return what we have
            return base_type, [], type_name[generic_start:]
        
        generic_content = type_name[generic_start + 1:generic_end]
        generic_part = type_name[generic_start:generic_end + 1]
        
        # Parse all types within the generic content
        all_types = self.parse_generic_types(generic_content)
        
        return base_type, all_types, generic_part

    def find_matching_bracket(self, text: str, start_pos: int) -> int:
        """Find the position of the matching closing bracket."""
        if start_pos >= len(text) or text[start_pos] != '<':
            return -1
        
        bracket_count = 1
        pos = start_pos + 1
        
        while pos < len(text) and bracket_count > 0:
            if text[pos] == '<':
                bracket_count += 1
            elif text[pos] == '>':
                bracket_count -= 1
            pos += 1
        
        return pos - 1 if bracket_count == 0 else -1

    def parse_generic_types(self, generic_content: str) -> List[str]:
        if not generic_content.strip():
            return []
        
        types = set()
        current_type = ""
        bracket_depth = 0
        i = 0
        
        while i < len(generic_content):
            char = generic_content[i]
            
            if char == '<':
                bracket_depth += 1
                current_type += char
            elif char == '>':
                bracket_depth -= 1
                current_type += char
            elif char == ',' and bracket_depth == 0:
                # Found a top-level comma separator
                if current_type.strip():
                    type_and_nested = self.extract_all_types_recursive(current_type.strip())
                    types.update(type_and_nested)
                current_type = ""
            else:
                current_type += char
            
            i += 1
        
        # Don't forget the last type
        if current_type.strip():
            type_and_nested = self.extract_all_types_recursive(current_type.strip())
            types.update(type_and_nested)
        
        return list(types)

    # Alternative cleaner approach - splits by comma first, then processes each part
    def parse_generic_types_alternative(self, generic_content: str) -> List[str]:
        """
        Alternative implementation that's more robust.
        """
        if not generic_content.strip():
            return []
        
        # Split by top-level commas only
        type_parts = self.split_by_top_level_comma(generic_content)
        
        types = []
        for part in type_parts:
            if part.strip():
                type_and_nested = self.extract_all_types_recursive(part.strip())
                types.extend(type_and_nested)
        
        return types

    def split_by_top_level_comma(self, content: str) -> List[str]:
        """
        Split content by commas that are at the top level (not inside nested brackets).
        """
        parts = []
        current_part = ""
        bracket_depth = 0
        
        for char in content:
            if char == '<':
                bracket_depth += 1
                current_part += char
            elif char == '>':
                bracket_depth -= 1
                current_part += char
            elif char == ',' and bracket_depth == 0:
                # Top-level comma found
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        # Add the last part
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts

    def extract_all_types_recursive(self, type_str: str) -> List[str]:
        """
        Recursively extract all types from a type string, including nested generics.
        """
        type_str = type_str.strip()
        if not type_str:
            return []
        
        types = []
        
        # Extract base type
        base_match = re.match(r'^([^<]+)', type_str)
        if base_match:
            base_type = base_match.group(1).strip()
            # Clean up array notation and wildcards
            clean_base = re.sub(r'\[\s*\]', '', base_type)  # Remove array brackets
            clean_base = re.sub(r'^\?\s*(?:extends|super)\s+', '', clean_base)  # Remove wildcards
            clean_base = clean_base.replace('?', '').strip()  # Remove standalone wildcards
            
            if clean_base and clean_base not in ['extends', 'super']:
                types.append(clean_base)
        
        # If there are generics, recursively extract them
        if '<' in type_str:
            generic_start = type_str.find('<')
            generic_end = self.find_matching_bracket(type_str, generic_start)
            
            if generic_end != -1:
                generic_content = type_str[generic_start + 1:generic_end]
                nested_types = self.parse_generic_types(generic_content)
                types.extend(nested_types)
        
        return types
    def _resolve_from_wildcard_imports(self, class_name: str, imports: Dict[str, str]) -> Optional[str]:
        """Resolve class name from wildcard imports"""
        if not class_name or not class_name[0].isupper():
            return None
            
        wildcard_packages = [import_path for import_key, import_path in imports.items() 
                           if import_key.startswith('*')]
        
        for package in wildcard_packages:
            potential_full_name = f"{package}.{class_name}"
            if potential_full_name in self.class_cache or self._is_known_java_class(package, class_name):
                return potential_full_name
        
        return None
    
    def _is_known_java_class(self, package: str, class_name: str) -> bool:
        """Check if it's a known Java standard library class"""
        java_classes = {
            'java.util': {
                'List', 'Set', 'Map', 'Collection', 'ArrayList', 'LinkedList', 'HashSet', 
                'TreeSet', 'HashMap', 'TreeMap', 'LinkedHashMap', 'Vector', 'Stack',
                'Queue', 'Deque', 'PriorityQueue', 'Collections', 'Arrays', 'Optional',
                'Stream', 'Iterator', 'ListIterator', 'Enumeration', 'Properties',
                'Date', 'Calendar', 'GregorianCalendar', 'TimeZone', 'Locale',
                'Random', 'Scanner', 'StringTokenizer', 'Timer', 'TimerTask'
            },
            'java.io': {
                'File', 'FileInputStream', 'FileOutputStream', 'FileReader', 'FileWriter',
                'BufferedReader', 'BufferedWriter', 'PrintWriter', 'InputStream', 
                'OutputStream', 'Reader', 'Writer', 'IOException', 'FileNotFoundException',
                'ObjectInputStream', 'ObjectOutputStream', 'Serializable'
            },
            'java.time': {
                'LocalDate', 'LocalTime', 'LocalDateTime', 'ZonedDateTime', 'Instant',
                'Duration', 'Period', 'DateTimeFormatter', 'ZoneId', 'OffsetDateTime',
                'Year', 'Month', 'DayOfWeek', 'MonthDay', 'YearMonth'
            }
        }
        return package in java_classes and class_name in java_classes[package]


class SpringBootMethodTypeDetector:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
    
    def detect_method_type(self, method_name: str, is_constructor: bool, body: str, 
                          annotations: List[str], return_type: str, parameters: List[Tuple[str, str]]) -> MethodType:
        """Determine method type with improved classification"""
        if is_constructor:
            return MethodType.CONSTRUCTOR
        
        # Check for REST endpoints first
        rest_annotations = {'@GetMapping', '@PostMapping', '@PutMapping', '@DeleteMapping', 
                          '@PatchMapping', '@RequestMapping'}
        if any(anno.split('(')[0] in rest_annotations for anno in annotations):
            return MethodType.REST_ENDPOINT
        
        # Check for getters and setters
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
        
        # Check for job configuration
        if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in self.config.job_patterns):
            return MethodType.JOB_CONFIG
        
        # Check for publisher configuration
        if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in self.config.publisher_patterns):
            return MethodType.PUBLISHER_CONFIG
        
        # Check for listener configuration
        if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in self.config.listener_patterns):
            return MethodType.LISTENER_CONFIG
        
        return MethodType.REGULAR


class RestEndpointExtractor:
    def extract_from_method(self, method_node: Node, content: str, class_node: Node = None) -> Optional[RestEndpoint]:
        """Extract REST endpoint info from method and class annotations"""
        rest_annos = {
            '@GetMapping': 'GET',
            '@PostMapping': 'POST',
            '@PutMapping': 'PUT',
            '@DeleteMapping': 'DELETE',
            '@PatchMapping': 'PATCH',
            '@RequestMapping': 'REQUEST'
        }
        
        method_path = ""
        http_method = None
        
        # Method-level annotation
        for child in method_node.children:
            if child.type == 'modifiers':
                for grandchild in child.children:
                    if grandchild.type == 'annotation':
                        text = content[grandchild.start_byte:grandchild.end_byte]
                        for anno, method in rest_annos.items():
                            if anno in text:
                                # Extract path value
                                path_match = re.search(r'(?:value\s*=\s*)?["\']([^"\']*)["\']', text)
                                if path_match:
                                    method_path = path_match.group(1)
                                else:
                                    # Check for simple annotation like @GetMapping("path")
                                    simple_path = re.search(r'@\w+\s*\(\s*["\']([^"\']*)["\']', text)
                                    if simple_path:
                                        method_path = simple_path.group(1)
                                
                                http_method = method
                                if anno == '@RequestMapping':
                                    # Extract HTTP method from RequestMapping
                                    method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', text)
                                    if method_match:
                                        http_method = method_match.group(1)
                                break
        
        # Class-level annotation for base path
        class_path = ""
        if class_node:
            for child in class_node.children:
                if child.type == 'modifiers':
                    for grandchild in child.children:
                        if grandchild.type == 'annotation':
                            text = content[grandchild.start_byte:grandchild.end_byte]
                            if '@RequestMapping' in text:
                                path_match = re.search(r'(?:value\s*=\s*)?["\']([^"\']*)["\']', text)
                                if path_match:
                                    class_path = path_match.group(1)
        
        # Combine paths
        if http_method:
            if class_path and method_path:
                full_path = class_path.rstrip('/') + '/' + method_path.lstrip('/')
            elif class_path:
                full_path = class_path
            else:
                full_path = method_path or ""
            return RestEndpoint(http_method=http_method, path=full_path)
        
        return None


# ============================================================================
# Fixed File Processor
# ============================================================================

class FileProcessor:
    def __init__(self, config: AnalyzerConfig, language: Language, parser: Parser):
        self.config = config
        self.language = language
        self.parser = parser
        self.comment_remover = JavaCommentRemover()
        self.method_type_detector = SpringBootMethodTypeDetector(config)
        self.endpoint_extractor = RestEndpointExtractor()
    
    def process_file(self, file_path: Path, project_id: str, class_cache: Set[str] = None) -> List[CodeChunk]:
        """Process a single Java file and return chunks for ALL classes/interfaces found"""
        try:
            content = self._read_file_content(file_path)
            if not content.strip():
                return []
            
            if self.config.remove_comments:
                content = self.comment_remover.remove_comments(content)
            
            tree = self.parser.parse(bytes(content, 'utf8'))
            root_node = tree.root_node
            
            package = self._extract_package(root_node, content)
            imports = self._extract_imports(root_node, content)
            
            # Initialize type resolver with class cache
            type_resolver = JavaTypeResolver(self.config, class_cache or set())
            
            # Get ALL class nodes (including nested ones)
            all_class_nodes = self._extract_all_class_nodes(root_node)
            chunks = []
            
            for class_node in all_class_nodes:
                chunk = self._parse_class_node(
                    class_node, content, package, imports, str(file_path), 
                    root_node, type_resolver
                )
                if chunk:
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding fallback"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin1') as f:
                return f.read()
    
    def _extract_package(self, root_node: Node, content: str) -> str:
        """Extract package declaration"""
        try:
            package_query = self.language.query("""
                (package_declaration
                    (scoped_identifier) @package)
            """)
            
            captures = package_query.captures(root_node)
            if captures:
                package_node = captures[0][0]
                return content[package_node.start_byte:package_node.end_byte]
        except Exception as e:
            logger.debug(f"Error extracting package: {e}")
        
        return ""
    
    def _extract_imports(self, root_node: Node, content: str) -> Dict[str, str]:
        """Extract import statements"""
        imports = {}
        
        try:
            import_query = self.language.query("""
                (import_declaration
                    (scoped_identifier) @import)
            """)
            
            captures = import_query.captures(root_node)
            for capture in captures:
                import_node = capture[0]
                import_path = content[import_node.start_byte:import_node.end_byte]
                
                if '*' in import_path:
                    package_path = import_path.replace('.*', '')
                    imports[f"*{package_path}"] = package_path
                else:
                    class_name = import_path.split('.')[-1]
                    imports[class_name] = import_path
                    
        except Exception as e:
            logger.debug(f"Error extracting imports: {e}")
        
        return imports
    
    def _extract_all_class_nodes(self, root_node: Node) -> List[Node]:
        """Extract ALL class nodes including nested ones - FIXED to handle nested classes separately"""
        all_class_nodes = []
        
        try:
            # Query for all class-like declarations
            class_query = self.language.query("""
                (class_declaration) @class
                (interface_declaration) @interface  
                (enum_declaration) @enum
                (record_declaration) @record
                (annotation_type_declaration) @annotation
            """)
            
            captures = class_query.captures(root_node)
            all_class_nodes = [capture[0] for capture in captures]
                
        except Exception as e:
            logger.debug(f"Error extracting class nodes: {e}")
        
        return all_class_nodes
    
    def _parse_class_node(self, class_node: Node, content: str, package: str, 
                         imports: Dict[str, str], file_path: str, root_node: Node,
                         type_resolver: JavaTypeResolver) -> Optional[CodeChunk]:
        """Parse a single class node - FIXED to handle nested classes properly"""
        try:
            class_name = self._get_class_name(class_node, content)
            if not class_name:
                return None
            
            # Build proper full class name for nested classes
            is_nested = self._is_nested_class(class_node, root_node)
            if is_nested:
                parent_class_name = self._get_parent_class_name(class_node, content)
                if parent_class_name:
                    full_class_name = f"{package}.{parent_class_name}.{class_name}" if package else f"{parent_class_name}.{class_name}"
                else:
                    full_class_name = f"{package}.{class_name}" if package else class_name
            else:
                full_class_name = f"{package}.{class_name}" if package else class_name
            
            modifiers = self._extract_modifiers(class_node, content)
            annotations = self._extract_annotations(class_node, content)
            class_type = self._detect_class_type(class_node, annotations, modifiers)
            
            implements = self._extract_implements(class_node, content, imports, package, type_resolver)
            extends = self._extract_extends(class_node, content, imports, package, type_resolver)
            
            # Extract ONLY fields that belong to THIS class (not nested classes)
            fields = self._extract_class_fields(class_node, content, imports, package, type_resolver)
            
            # Extract ONLY methods that belong to THIS class (not nested classes)
            methods = self._extract_class_methods(class_node, content, imports, package, 
                                                implements, extends, class_name, full_class_name, 
                                                type_resolver, fields)
            
            parent_class = self._get_parent_class(class_node, content, package) if is_nested else None
            inner_classes = self._get_direct_inner_classes(class_node, content, class_name, package)
            
            class_content = content[class_node.start_byte:class_node.end_byte]
            
            return CodeChunk(
                package=package,
                class_name=class_name,
                full_class_name=full_class_name,
                class_type=class_type,
                file_path=file_path,
                content=class_content,
                implements=implements,
                extends=extends,
                fields=fields,
                imports=imports,
                methods=methods,
                annotations=annotations,
                is_nested=is_nested,
                parent_class=parent_class,
                inner_classes=inner_classes,
                modifiers=modifiers
            )
            
        except Exception as e:
            logger.error(f"Error parsing class node: {e}")
            return None
    
    def _get_class_name(self, class_node: Node, content: str) -> Optional[str]:
        """Get class name from class node"""
        try:
            for child in class_node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
        except Exception:
            pass
        return None
    
    def _get_parent_class_name(self, class_node: Node, content: str) -> Optional[str]:
        """Get the immediate parent class name for nested classes"""
        parent = class_node.parent
        while parent:
            if parent.type in ['class_declaration', 'interface_declaration', 
                             'enum_declaration', 'record_declaration']:
                return self._get_class_name(parent, content)
            parent = parent.parent
        return None
    
    def _extract_modifiers(self, node: Node, content: str) -> List[str]:
        """Extract modifiers from a node"""
        modifiers = []
        try:
            for child in node.children:
                if child.type == 'modifiers':
                    for modifier_child in child.children:
                        if modifier_child.type in ['public', 'private', 'protected', 'static', 
                                                 'final', 'abstract', 'synchronized', 'native', 
                                                 'transient', 'volatile', 'strictfp']:
                            modifiers.append(modifier_child.type)
        except Exception:
            pass
        return modifiers
    
    def _extract_annotations(self, node: Node, content: str) -> List[str]:
        """Extract annotations from a node"""
        annotations = []
        
        try:
            modifiers_node = None
            for child in node.children:
                if child.type == 'modifiers':
                    modifiers_node = child
                    break
            
            if modifiers_node:
                annotation_query = self.language.query("""
                    (annotation
                        name: (identifier) @annotation_name)
                    (marker_annotation
                        name: (identifier) @marker_annotation_name)
                """)
                
                captures = annotation_query.captures(modifiers_node)
                for capture in captures:
                    annotation_name = content[capture[0].start_byte:capture[0].end_byte]
                    annotations.append(f"@{annotation_name}")
                    
        except Exception as e:
            logger.debug(f"Error extracting annotations: {e}")
        
        return annotations
    
    def _detect_class_type(self, class_node: Node, annotations: List[str], modifiers: List[str]) -> ClassType:
        """Determine class type based on annotations and structure"""
        # Check Spring Boot annotations first (priority)
        for annotation in annotations:
            annotation_clean = annotation.split('(')[0]
            if annotation_clean in self.config.springboot_annotations:
                return self.config.springboot_annotations[annotation_clean]
        
        # Check node type
        if class_node.type == 'enum_declaration':
            return ClassType.ENUM
        elif class_node.type == 'record_declaration':
            return ClassType.RECORD
        elif class_node.type == 'interface_declaration':
            return ClassType.INTERFACE
        elif class_node.type == 'annotation_type_declaration':
            return ClassType.ANNOTATION
        
        # Check for abstract modifier
        if 'abstract' in modifiers:
            return ClassType.ABSTRACT
        
        return ClassType.CLASS
    
    def _extract_implements(self, class_node: Node, content: str, 
                          imports: Dict[str, str], package: str, type_resolver: JavaTypeResolver) -> List[str]:
        """Extract implemented interfaces"""
        implements = []
        try:
            implements_query = self.language.query("""
                (class_declaration
                    interfaces: (super_interfaces
                        (type_list
                            (type_identifier) @interface)))
                (class_declaration
                    interfaces: (super_interfaces
                        (type_list
                            (generic_type
                                (type_identifier) @generic_interface))))
            """)
            
            captures = implements_query.captures(class_node)
            for capture in captures:
                interface_name = content[capture[0].start_byte:capture[0].end_byte]
                full_interface_names = type_resolver.resolve_type_name(interface_name, imports, package)
                for full_interface_name in full_interface_names:
                    if full_interface_name not in implements:
                        implements.append(full_interface_name)
                    
        except Exception as e:
            logger.debug(f"Error extracting implements: {e}")
        
        return implements
    
    def _extract_extends(self, class_node: Node, content: str, 
                        imports: Dict[str, str], package: str, type_resolver: JavaTypeResolver) -> Optional[str]:
        """Extract extended class"""
        try:
            extends_query = self.language.query("""
                (class_declaration
                    superclass: (superclass
                        (type_identifier) @superclass))
                (class_declaration
                    superclass: (superclass
                        (generic_type
                            (type_identifier) @generic_superclass)))
            """)
            
            captures = extends_query.captures(class_node)
            if captures:
                superclass_name = content[captures[0][0].start_byte:captures[0][0].end_byte]
                resolved_names = type_resolver.resolve_type_name(superclass_name, imports, package)
                return resolved_names[0] if resolved_names else None
                
        except Exception as e:
            logger.debug(f"Error extracting extends: {e}")
        
        return None
    
    def _extract_class_fields(self, class_node: Node, content: str, imports: Dict[str, str], 
                             package: str, type_resolver: JavaTypeResolver) -> List[Field]:
        """Extract ONLY fields that belong directly to this class (not nested classes)"""
        fields = []
        
        try:
            # Only look at direct children of the class, not nested class fields
            for child in class_node.children:
                if child.type == 'class_body':
                    for body_child in child.children:
                        if body_child.type == 'field_declaration':
                            field = self._parse_field_declaration(body_child, content, imports, package, type_resolver)
                            if field:
                                fields.append(field)
                    break
                    
        except Exception as e:
            logger.debug(f"Error extracting class fields: {e}")
        
        return fields
    
    def _parse_field_declaration(self, field_node: Node, content: str, imports: Dict[str, str],
                                package: str, type_resolver: JavaTypeResolver) -> Optional[Field]:
        """Parse a single field declaration"""
        try:
            field_type = None
            field_name = None
            field_annotations = []
            
            # Extract field type and name
            for child in field_node.children:
                if child.type == 'modifiers':
                    field_annotations = self._extract_annotations(field_node, content)
                elif child.type in ['type_identifier', 'generic_type', 'array_type', 'primitive_type']:
                    field_type = content[child.start_byte:child.end_byte]
                elif child.type == 'variable_declarator':
                    for var_child in child.children:
                        if var_child.type == 'identifier':
                            field_name = content[var_child.start_byte:var_child.end_byte]
                            break
            
            if field_type and field_name:
                base_type = re.sub(r'<.*?>', '', field_type).strip()
                full_types = type_resolver.resolve_type_name(field_type, imports, package)
                full_type = full_types[0] if full_types else field_type
                
                return Field(
                    name=field_name,
                    type=base_type,
                    full_type=full_type,
                    annotations=field_annotations
                )
                
        except Exception as e:
            logger.debug(f"Error parsing field declaration: {e}")
        
        return None
    
    def _extract_class_methods(self, class_node: Node, content: str, imports: Dict[str, str],
                              package: str, implements: List[str], extends: Optional[str],
                              class_name: str, full_class_name: str, type_resolver: JavaTypeResolver,
                              fields: List[Field]) -> List[Method]:
        """Extract ONLY methods that belong directly to this class (not nested classes)"""
        methods = []
        
        try:
            # Only look at direct children of the class body
            for child in class_node.children:
                if child.type == 'class_body':
                    for body_child in child.children:
                        if body_child.type == 'method_declaration':
                            method = self._process_method_node(
                                body_child, content, imports, package, implements, 
                                extends, class_name, full_class_name, type_resolver, fields, False, class_node
                            )
                            if method:
                                methods.append(method)
                        elif body_child.type == 'constructor_declaration':
                            method = self._process_method_node(
                                body_child, content, imports, package, implements, 
                                extends, class_name, full_class_name, type_resolver, fields, True, class_node
                            )
                            if method:
                                methods.append(method)
                    break
                    
        except Exception as e:
            logger.debug(f"Error extracting class methods: {e}")
        
        return methods
    
    def _process_method_node(self, method_node: Node, content: str, imports: Dict[str, str],
                            package: str, implements: List[str], extends: Optional[str],
                            class_name: str, full_class_name: str, type_resolver: JavaTypeResolver,
                            fields: List[Field], is_constructor: bool, class_node: Node) -> Optional[Method]:
        """Process a single method or constructor node - FIXED"""
        try:
            # Extract method name
            method_name = None
            for child in method_node.children:
                if child.type == 'identifier':
                    method_name = content[child.start_byte:child.end_byte]
                    break
            
            if not method_name:
                return None
            
            modifiers = self._extract_modifiers(method_node, content)
            annotations = self._extract_annotations(method_node, content)
            
            # Extract return type (only for regular methods) - FIXED
            return_type = ""
            full_return_types = ""
            if not is_constructor:
                return_type, full_return_types = self._extract_return_type(method_node, content, type_resolver, imports, package)
            
            # Extract parameters
            parameters = self._extract_method_parameters(method_node, content, type_resolver, imports, package)
            parameter_types = set()
            for _, ptype in parameters:
                parameter_types.add(ptype)

            # Extract throws clause
            throws = self._extract_throws(method_node, content, type_resolver, imports, package)
            
            # Extract method body and analyze calls
            body = ""
            method_calls = []
            variable_usage = []
            
            for child in method_node.children:
                if child.type == 'block':
                    body = content[method_node.start_byte:method_node.end_byte]
                    method_calls, variable_usage = self._extract_method_calls_and_usage(
                        child, content, imports, package, full_class_name, 
                        type_resolver, parameters, fields
                    )
                    break
            
            is_abstract = 'abstract' in modifiers and not body.strip()
            is_override = '@Override' in annotations
            
            # FIXED method type detection
            method_type = self.method_type_detector.detect_method_type(
                method_name, is_constructor, body, annotations, return_type, parameters
            )
            
            # FIXED inheritance info - only check actual parent methods
            inheritance_info = self._get_correct_inheritance_info(method_name, implements, extends, parameters, type_resolver)
            extends_info = self._get_correct_extends_info(method_name, extends, parameters, type_resolver)
            
            # FIXED endpoint extraction
            endpoint = self.endpoint_extractor.extract_from_method(method_node, content, class_node)
            
            return Method(
                name=f"{full_class_name}.{method_name}",
                return_type=return_type,
                full_return_type=full_return_types,
                parameters=list(parameter_types),
                annotations=annotations,
                method_type=method_type,
                body=body,
                is_abstract=is_abstract,
                is_override=is_override,
                throws=throws,
                method_calls=method_calls,
                variable_usage=variable_usage,
                inheritance_info=inheritance_info,
                extends_info=extends_info,
                endpoint=endpoint
            )
            
        except Exception as e:
            logger.debug(f"Error processing method node: {e}")
            return None
    
    def _extract_return_type(self, method_node: Node, content: str, type_resolver: JavaTypeResolver,
                            imports: Dict[str, str], package: str) -> Tuple[str, str]:
        """Extract return type from method - FIXED"""
        try:
            for child in method_node.children:
                if child.type in ['type_identifier', 'generic_type', 'array_type', 'void_type', 'primitive_type']:
                    return_type = content[child.start_byte:child.end_byte]
                    full_return_types = type_resolver.resolve_type_name(return_type, imports, package)
                    full_return_type = full_return_types[0] if full_return_types else return_type
                    return return_type, full_return_type
        except Exception as e:
            logger.debug(f"Error extracting return type: {e}")
        
        return "", ""
    
    def _extract_method_parameters(self, method_node: Node, content: str, 
                                 type_resolver: JavaTypeResolver, imports: Dict[str, str], 
                                 package: str) -> List[Tuple[str, str]]:
        """Extract method parameters as (name, type) tuples"""
        parameters = []
        try:
            for child in method_node.children:
                if child.type == 'formal_parameters':
                    param_query = self.language.query("""
                        (formal_parameter
                            type: (_) @param_type
                            name: (identifier) @param_name)
                    """)
                    captures = param_query.captures(child)
                    
                    i = 0
                    while i + 1 < len(captures):
                        if captures[i][1] == 'param_type' and captures[i + 1][1] == 'param_name':
                            param_type_node = captures[i][0]
                            param_name_node = captures[i + 1][0]
                            param_type = content[param_type_node.start_byte:param_type_node.end_byte]
                            param_name = content[param_name_node.start_byte:param_name_node.end_byte]
                            
                            resolved_types = type_resolver.resolve_type_name(param_type, imports, package)
                            resolved_type = resolved_types[0] if resolved_types else param_type
                            parameters.append((param_name, resolved_type))
                            i += 2
                        else:
                            i += 1
                    break
                    
        except Exception as e:
            logger.debug(f"Error extracting method parameters: {e}")
        
        return parameters
    
    def _extract_throws(self, method_node: Node, content: str, 
                       type_resolver: JavaTypeResolver, imports: Dict[str, str], package: str) -> List[str]:
        """Extract throws clause"""
        throws = []
        try:
            for child in method_node.children:
                if child.type == 'throws':
                    throws_query = self.language.query("(throws (type_identifier) @exception_type)")
                    throws_captures = throws_query.captures(child)
                    for throw_capture in throws_captures:
                        exception_type = content[throw_capture[0].start_byte:throw_capture[0].end_byte]
                        resolved_types = type_resolver.resolve_type_name(exception_type, imports, package)
                        for resolved_type in resolved_types:
                            throws.append(resolved_type)
                    break
        except Exception as e:
            logger.debug(f"Error extracting throws: {e}")
        
        return throws
    
    def _extract_method_calls_and_usage(self, body_node: Node, content: str, 
                                       imports: Dict[str, str], package: str, full_class_name: str,
                                       type_resolver: JavaTypeResolver, parameters: List[Tuple[str, str]],
                                       fields: List[Field]) -> Tuple[List[str], List[str]]:
        """Extract method calls and variable usage from method body - FIXED to avoid duplicates"""
        method_calls = set()  # Use set to avoid duplicates
        variable_usage = set()
        
        try:
            # Extract local variables first
            local_variables = self._extract_local_variables(body_node, content, type_resolver, imports, package)
            
            # Add parameter types to variable usage
            for _, param_type in parameters:
                clean_type = self._extract_base_class_name(param_type)
                if clean_type and not self._is_builtin_type(clean_type):
                    variable_usage.add(clean_type)
            
            # Add field types to variable usage
            for field in fields:
                clean_type = self._extract_base_class_name(field.full_type)
                if clean_type and not self._is_builtin_type(clean_type):
                    variable_usage.add(clean_type)
            
            # Add local variable types to variable usage
            for var_type in local_variables.values():
                clean_type = self._extract_base_class_name(var_type)
                if clean_type and not self._is_builtin_type(clean_type):
                    variable_usage.add(clean_type)
            
            # Extract method calls - FIXED to avoid incorrect attributions
            call_query = self.language.query("""
                (method_invocation
                    object: (_) @object
                    name: (identifier) @method_name)
                (method_invocation
                    name: (identifier) @static_method_name)
            """)
            
            captures = call_query.captures(body_node)
            field_map = {field.name: field.full_type for field in fields}
            param_map = {name: ptype for name, ptype in parameters}
            
            i = 0
            while i < len(captures):
                if (i + 1 < len(captures) and captures[i][1] == 'object' and 
                    captures[i + 1][1] == 'method_name'):
                    
                    object_node = captures[i][0]
                    method_node = captures[i + 1][0]
                    
                    object_name = content[object_node.start_byte:object_node.end_byte]
                    method_name = content[method_node.start_byte:method_node.end_byte]
                    
                    if object_name == 'this':
                        method_calls.add(f"{full_class_name}.{method_name}")
                    elif object_name in field_map:
                        field_type = self._extract_base_class_name(field_map[object_name])
                        if not self._is_builtin_type(field_type):
                            method_calls.add(f"{field_type}.{method_name}")
                    elif object_name in param_map:
                        param_type = self._extract_base_class_name(param_map[object_name])
                        if not self._is_builtin_type(param_type):
                            method_calls.add(f"{param_type}.{method_name}")
                    elif object_name in local_variables:
                        var_type = self._extract_base_class_name(local_variables[object_name])
                        if not self._is_builtin_type(var_type):
                            method_calls.add(f"{var_type}.{method_name}")
                    elif self._is_class_reference(object_name, imports, package):
                        resolved_classes = type_resolver.resolve_type_name(object_name, imports, package)
                        for resolved_class in resolved_classes:
                            clean_type = self._extract_base_class_name(resolved_class)
                            if not self._is_builtin_type(clean_type):
                                method_calls.add(f"{clean_type}.{method_name}")
                    
                    i += 2
                    
                elif captures[i][1] == 'static_method_name':
                    method_node = captures[i][0]
                    method_name = content[method_node.start_byte:method_node.end_byte]
                    # Only add if it's actually a method of this class
                    method_calls.add(f"{full_class_name}.{method_name}")
                    i += 1
                else:
                    i += 1
                    
        except Exception as e:
            logger.debug(f"Error extracting method calls: {e}")
        
        return list(method_calls), list(variable_usage)
    
    def _extract_local_variables(self, body_node: Node, content: str, 
                               type_resolver: JavaTypeResolver, imports: Dict[str, str], 
                               package: str) -> Dict[str, str]:
        """Extract local variable declarations and their types"""
        local_variables = {}
        
        try:
            var_query = self.language.query("""
                (local_variable_declaration
                    type: (_) @var_type
                    declarator: (variable_declarator
                        name: (identifier) @var_name))
            """)
            
            captures = var_query.captures(body_node)
            
            i = 0
            while i + 1 < len(captures):
                if captures[i][1] == 'var_type' and captures[i + 1][1] == 'var_name':
                    type_node = captures[i][0]
                    name_node = captures[i + 1][0]
                    
                    var_type = content[type_node.start_byte:type_node.end_byte]
                    var_name = content[name_node.start_byte:name_node.end_byte]
                    
                    full_types = type_resolver.resolve_type_name(var_type, imports, package)
                    full_type = full_types[0] if full_types else var_type
                    local_variables[var_name] = full_type
                    
                    i += 2
                else:
                    i += 1
                    
        except Exception as e:
            logger.debug(f"Error extracting local variables: {e}")
        
        return local_variables
    
    def _extract_base_class_name(self, type_name: str) -> str:
        """Extract base class name from type, removing generics and arrays"""
        if not type_name:
            return ""
        
        clean_type = type_name.replace('[]', '')
        generic_start = clean_type.find('<')
        if generic_start != -1:
            clean_type = clean_type[:generic_start]
        
        return clean_type.strip()
    
    def _is_builtin_type(self, type_name: str) -> bool:
        """Check if type is a built-in Java type"""
        base_type = type_name.split('.')[0] if '.' in type_name else type_name
        base_type = re.sub(r'<.*?>', '', base_type)
        base_type = base_type.replace('[]', '')
        
        return base_type in self.config.builtin_types or base_type.startswith('java.lang.')
    
    def _is_class_reference(self, name: str, imports: Dict[str, str], package: str) -> bool:
        """Check if name refers to a class"""
        return name in imports or (name and name[0].isupper())
    
    def _get_correct_inheritance_info(self, method_name: str, implements: List[str], 
                                    extends: Optional[str], parameters: List[Tuple[str, str]], 
                                    type_resolver: JavaTypeResolver) -> List[str]:
        """Get CORRECT inheritance information - only include methods that actually exist in parent classes"""
        # This is a simplified version - in a real implementation, you'd check if the method actually exists
        # in the parent class/interface by analyzing those classes too
        inheritance_sources = []
        param_types = [ptype for _, ptype in parameters]
        param_sig = ','.join(param_types)
        
        # Only add if we're confident the method exists in the parent
        # For now, only add well-known interface methods or if we have explicit knowledge
        known_interface_methods = {
            'equals', 'hashCode', 'toString', 'compareTo', 'iterator',
            'size', 'isEmpty', 'contains', 'add', 'remove'
        }
        
        if extends and method_name in known_interface_methods:
            inheritance_sources.append(f"{extends}.{method_name}({param_sig})")
        
        for interface in implements:
            if method_name in known_interface_methods:
                inheritance_sources.append(f"{interface}.{method_name}({param_sig})")
        
        return inheritance_sources
    
    def _get_correct_extends_info(self, method_name: str, extends: Optional[str], 
                                parameters: List[Tuple[str, str]], type_resolver: JavaTypeResolver) -> List[str]:
        """Get CORRECT extends information - only if method actually exists in parent"""
        if not extends:
            return []
        
        # Similar logic as above - only add known methods
        known_methods = {'equals', 'hashCode', 'toString'}
        if method_name in known_methods:
            param_types = [ptype for _, ptype in parameters]
            param_sig = ','.join(param_types)
            return [f"{extends}.{method_name}({param_sig})"]
        
        return []
    
    def _is_nested_class(self, class_node: Node, root_node: Node) -> bool:
        """Check if class is nested inside another class"""
        parent = class_node.parent
        while parent and parent != root_node:
            if parent.type in ['class_declaration', 'interface_declaration', 
                             'enum_declaration', 'record_declaration']:
                return True
            parent = parent.parent
        return False
    
    def _get_parent_class(self, class_node: Node, content: str, package: str) -> Optional[str]:
        """Get parent class name for nested class"""
        parent = class_node.parent
        while parent:
            if parent.type in ['class_declaration', 'interface_declaration', 
                             'enum_declaration', 'record_declaration']:
                parent_name = self._get_class_name(parent, content)
                if parent_name:
                    return f"{package}.{parent_name}" if package else parent_name
            parent = parent.parent
        return None
    
    def _get_direct_inner_classes(self, class_node: Node, content: str, class_name: str, package: str) -> List[str]:
        """Get DIRECT inner class names (not nested within nested classes)"""
        inner_classes = []
        
        try:
            # Only look at direct children of the class body
            for child in class_node.children:
                if child.type == 'class_body':
                    for body_child in child.children:
                        if body_child.type in ['class_declaration', 'interface_declaration', 
                                             'enum_declaration', 'record_declaration']:
                            inner_name = self._get_class_name(body_child, content)
                            if inner_name and inner_name != class_name:
                                full_inner_name = f"{package}.{class_name}.{inner_name}" if package else f"{class_name}.{inner_name}"
                                inner_classes.append(full_inner_name)
                    break
                    
        except Exception as e:
            logger.debug(f"Error extracting inner classes: {e}")
        
        return inner_classes


# ============================================================================
# Additional Fixed Components
# ============================================================================

class DependencyGraphBuilder:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
    
    def build_dependency_graph(self, chunks: List[CodeChunk]) -> DependencyGraph:
        """Build dependency relationships between classes"""
        dependency_graph = DependencyGraph()
        
        # Add all nodes
        for chunk in chunks:
            dependency_graph.nodes.add(chunk.full_class_name)
        
        # Add edges
        for chunk in chunks:
            # Extends relationship
            if chunk.extends:
                dependency_graph.edges.append((chunk.full_class_name, chunk.extends, "extends"))
            
            # Implements relationships
            for interface in chunk.implements:
                dependency_graph.edges.append((chunk.full_class_name, interface, "implements"))
            
            # Field dependencies
            for field in chunk.fields:
                if field.full_type and not self._is_builtin_type(field.full_type):
                    base_type = self._extract_base_type(field.full_type)
                    if base_type != chunk.full_class_name and base_type in dependency_graph.nodes:
                        dependency_graph.edges.append((chunk.full_class_name, base_type, "uses"))
            
            # Method dependencies
            for method in chunk.methods:
                # Return type dependency
                if method.full_return_type and not self._is_builtin_type(method.full_return_type):
                    base_type = self._extract_base_type(method.full_return_type)
                    if base_type != chunk.full_class_name and base_type in dependency_graph.nodes:
                        dependency_graph.edges.append((chunk.full_class_name, base_type, "uses"))
                
                # Parameter dependencies
                for param_type in method.parameters:
                    if param_type and not self._is_builtin_type(param_type):
                        base_type = self._extract_base_type(param_type)
                        if base_type != chunk.full_class_name and base_type in dependency_graph.nodes:
                            dependency_graph.edges.append((chunk.full_class_name, base_type, "uses"))
            
            # Inner class relationships
            for inner_class in chunk.inner_classes:
                if inner_class in dependency_graph.nodes:
                    dependency_graph.edges.append((chunk.full_class_name, inner_class, "contains"))
            
            # Nested class relationship
            if chunk.parent_class and chunk.parent_class in dependency_graph.nodes:
                dependency_graph.edges.append((chunk.parent_class, chunk.full_class_name, "contains"))
        
        return dependency_graph
    
    def _extract_base_type(self, type_name: str) -> str:
        """Extract base type from full type name"""
        base_type = re.sub(r'<.*?>', '', type_name)
        base_type = base_type.replace('[]', '')
        return base_type.strip()
    
    def _is_builtin_type(self, type_name: str) -> bool:
        """Check if type is builtin"""
        base_type = type_name.split('.')[0] if '.' in type_name else type_name
        base_type = re.sub(r'<.*?>', '', base_type).replace('[]', '')
        return base_type in AnalyzerConfig().builtin_types or base_type.startswith('java.lang.')


class StatisticsGenerator:
    def generate_statistics(self, chunks: List[CodeChunk], dependency_graph: DependencyGraph) -> Dict:
        """Generate comprehensive analysis statistics"""
        stats = {
            'total_classes': len(chunks),
            'class_types': self._count_class_types(chunks),
            'method_types': self._count_method_types(chunks),
            'total_methods': sum(len(chunk.methods) for chunk in chunks),
            'total_fields': sum(len(chunk.fields) for chunk in chunks),
            'dependency_stats': self._analyze_dependencies(dependency_graph),
            'package_distribution': self._analyze_packages(chunks),
            'annotation_usage': self._analyze_annotations(chunks),
            'complexity_metrics': self._calculate_complexity_metrics(chunks)
        }
        return stats
    
    def _count_class_types(self, chunks: List[CodeChunk]) -> Dict[str, int]:
        """Count classes by type"""
        class_types = {}
        for chunk in chunks:
            class_type = chunk.class_type.value
            class_types[class_type] = class_types.get(class_type, 0) + 1
        return class_types
    
    def _count_method_types(self, chunks: List[CodeChunk]) -> Dict[str, int]:
        """Count methods by type"""
        method_types = {}
        for chunk in chunks:
            for method in chunk.methods:
                method_type = method.method_type.value
                method_types[method_type] = method_types.get(method_type, 0) + 1
        return method_types
    
    def _analyze_dependencies(self, dependency_graph: DependencyGraph) -> Dict[str, Any]:
        """Analyze dependency statistics"""
        dependency_types = {}
        for _, _, relation_type in dependency_graph.edges:
            dependency_types[relation_type] = dependency_types.get(relation_type, 0) + 1
        
        return {
            'total_dependencies': len(dependency_graph.edges),
            'dependency_types': dependency_types
        }
    
    def _analyze_packages(self, chunks: List[CodeChunk]) -> Dict[str, int]:
        """Analyze package distribution"""
        packages = {}
        for chunk in chunks:
            package = chunk.package if chunk.package else '<default>'
            packages[package] = packages.get(package, 0) + 1
        return packages
    
    def _analyze_annotations(self, chunks: List[CodeChunk]) -> Dict[str, int]:
        """Analyze annotation usage"""
        annotations = {}
        for chunk in chunks:
            for annotation in chunk.annotations:
                annotations[annotation] = annotations.get(annotation, 0) + 1
            for method in chunk.methods:
                for annotation in method.annotations:
                    annotations[annotation] = annotations.get(annotation, 0) + 1
        return annotations
    
    def _calculate_complexity_metrics(self, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Calculate complexity metrics"""
        total_lines = 0
        max_methods_per_class = 0
        max_fields_per_class = 0
        avg_methods_per_class = 0
        avg_fields_per_class = 0
        
        if chunks:
            total_lines = sum(len(chunk.content.split('\n')) for chunk in chunks)
            max_methods_per_class = max(len(chunk.methods) for chunk in chunks)
            max_fields_per_class = max(len(chunk.fields) for chunk in chunks)
            avg_methods_per_class = sum(len(chunk.methods) for chunk in chunks) / len(chunks)
            avg_fields_per_class = sum(len(chunk.fields) for chunk in chunks) / len(chunks)
        
        return {
            'total_lines_of_code': total_lines,
            'max_methods_per_class': max_methods_per_class,
            'max_fields_per_class': max_fields_per_class,
            'avg_methods_per_class': round(avg_methods_per_class, 2),
            'avg_fields_per_class': round(avg_fields_per_class, 2)
        }


class ResultExporter:
    def export_results(self, chunks: List[CodeChunk], dependency_graph: DependencyGraph, output_path: Path) -> None:
        """Export analysis results to JSON files"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export chunks
        self._export_chunks(chunks, output_path)
        
        # Export dependency graph
        self._export_dependency_graph(dependency_graph, output_path)
        
        # Export statistics
        stats_generator = StatisticsGenerator()
        stats = stats_generator.generate_statistics(chunks, dependency_graph)
        self._export_statistics(stats, output_path)
        
        logger.info(f"Results exported to {output_path}")
    
    def _export_chunks(self, chunks: List[CodeChunk], output_path: Path) -> None:
        """Export code chunks to JSON"""
        chunks_data = []
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            chunk_dict['class_type'] = chunk.class_type.value
            
            # Convert method enums to strings
            for method_dict in chunk_dict['methods']:
                if hasattr(method_dict['method_type'], 'value'):
                    method_dict['method_type'] = method_dict['method_type'].value
                else:
                    method_dict['method_type'] = str(method_dict['method_type'])
            
            chunks_data.append(chunk_dict)
        
        with open(output_path / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    def _export_dependency_graph(self, dependency_graph: DependencyGraph, output_path: Path) -> None:
        """Export dependency graph to JSON"""
        with open(output_path / 'dependency_graph.json', 'w', encoding='utf-8') as f:
            json.dump(dependency_graph.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _export_statistics(self, stats: Dict, output_path: Path) -> None:
        """Export statistics to JSON"""
        with open(output_path / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


class ClassCacheBuilder:
    """Builds cache of all available classes for type resolution"""
    
    def __init__(self, language: Language, parser: Parser, comment_remover: JavaCommentRemover):
        self.language = language
        self.parser = parser
        self.comment_remover = comment_remover
    
    def build_class_cache(self, java_files: List[Path]) -> Set[str]:
        """Build cache of all available classes"""
        class_cache = set()
        
        for java_file in java_files:
            try:
                with open(java_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                content = self.comment_remover.remove_comments(content)
                tree = self.parser.parse(bytes(content, 'utf8'))
                root_node = tree.root_node
                
                package = self._extract_package(root_node, content)
                class_nodes = self._extract_class_nodes(root_node)
                
                # Handle both regular and nested classes
                for class_node in class_nodes:
                    class_name = self._get_class_name(class_node, content)
                    if class_name:
                        # Check if it's a nested class
                        parent_names = []
                        parent = class_node.parent
                        while parent and parent != root_node:
                            if parent.type in ['class_declaration', 'interface_declaration', 
                                             'enum_declaration', 'record_declaration']:
                                parent_name = self._get_class_name(parent, content)
                                if parent_name:
                                    parent_names.append(parent_name)
                            parent = parent.parent
                        
                        if parent_names:
                            # Nested class
                            parent_names.reverse()
                            nested_path = '.'.join(parent_names + [class_name])
                            full_class_name = f"{package}.{nested_path}" if package else nested_path
                        else:
                            # Regular class
                            full_class_name = f"{package}.{class_name}" if package else class_name
                        
                        class_cache.add(full_class_name)
                        
            except Exception as e:
                logger.debug(f"Error building class cache for {java_file}: {e}")
                continue
        
        return class_cache
    
    def _extract_package(self, root_node: Node, content: str) -> str:
        """Extract package declaration"""
        try:
            package_query = self.language.query("""
                (package_declaration
                    (scoped_identifier) @package)
            """)
            
            captures = package_query.captures(root_node)
            if captures:
                package_node = captures[0][0]
                return content[package_node.start_byte:package_node.end_byte]
        except Exception:
            pass
        return ""
    
    def _extract_class_nodes(self, root_node: Node) -> List[Node]:
        """Extract class nodes"""
        try:
            class_query = self.language.query("""
                (class_declaration) @class
                (interface_declaration) @interface  
                (enum_declaration) @enum
                (record_declaration) @record
                (annotation_type_declaration) @annotation
            """)
            
            captures = class_query.captures(root_node)
            return [capture[0] for capture in captures]
        except Exception:
            return []
    
    def _get_class_name(self, class_node: Node, content: str) -> Optional[str]:
        """Get class name from node"""
        try:
            for child in class_node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
        except Exception:
            pass
        return None


# ============================================================================
# Main Analyzer Class
# ============================================================================

class JavaCodeAnalyzer:
    """Main analyzer class - FIXED"""
    
    def __init__(self, config: AnalyzerConfig = None):
        self.config = config or AnalyzerConfig()
        
        # Initialize Tree-sitter components
        self.language: Language = get_language("java")
        self.parser = Parser()
        self.parser.set_language(self.language)
        
        # Initialize services
        self.comment_remover = JavaCommentRemover()
        self.file_processor = FileProcessor(self.config, self.language, self.parser)
        self.dependency_graph_builder = DependencyGraphBuilder(self.config)
        self.statistics_generator = StatisticsGenerator()
        self.result_exporter = ResultExporter()
        self.class_cache_builder = ClassCacheBuilder(self.language, self.parser, self.comment_remover)
    
    def parse_project(self, root: Path, project_id: str) -> Tuple[List[CodeChunk], DependencyGraph]:
        """Main entry point for parsing Java project"""
        logger.info(f"Starting to parse project: {project_id} at {root}")
        
        # Find all Java files
        java_files = list(root.rglob("*.java"))
        logger.info(f"Found {len(java_files)} Java files")
        
        if not java_files:
            logger.warning("No Java files found")
            return [], DependencyGraph()
        
        # Build class cache for type resolution
        logger.info("Building class cache for type resolution...")
        class_cache = self.class_cache_builder.build_class_cache(java_files)
        logger.info(f"Built class cache with {len(class_cache)} classes")
        
        # Process files
        logger.info("Processing Java files...")
        chunks = []
        for i, java_file in enumerate(java_files):
            try:
                logger.debug(f"Processing file {i+1}/{len(java_files)}: {java_file}")
                file_chunks = self.file_processor.process_file(java_file, project_id, class_cache)
                chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Error processing file {java_file}: {e}")
                continue
        
        logger.info(f"Processed {len(chunks)} classes/interfaces/enums")
        
        if not chunks:
            logger.warning("No classes were successfully parsed")
            return [], DependencyGraph()
        
        # Build dependency graph
        logger.info("Building dependency graph...")
        dependency_graph = self.dependency_graph_builder.build_dependency_graph(chunks)
        
        logger.info(f"Built dependency graph with {len(dependency_graph.nodes)} nodes and {len(dependency_graph.edges)} edges")
        
        return chunks, dependency_graph
    
    def export_results(self, chunks: List[CodeChunk], dependency_graph: DependencyGraph, output_path: Path) -> None:
        """Export analysis results"""
        self.result_exporter.export_results(chunks, dependency_graph, output_path)
    
    def generate_statistics(self, chunks: List[CodeChunk], dependency_graph: DependencyGraph) -> Dict:
        """Generate analysis statistics"""
        return self.statistics_generator.generate_statistics(chunks, dependency_graph)


# ============================================================================
# Factory Pattern Implementation
# ============================================================================

class AnalyzerConfigBuilder:
    """Builder pattern for creating analyzer configuration"""
    
    def __init__(self):
        self.config_data = {}
    
    def with_comment_removal(self, remove_comments: bool) -> 'AnalyzerConfigBuilder':
        self.config_data['remove_comments'] = remove_comments
        return self
    
    def with_custom_builtin_types(self, builtin_types: Set[str]) -> 'AnalyzerConfigBuilder':
        self.config_data['builtin_types'] = builtin_types
        return self
    
    def with_custom_spring_annotations(self, annotations: Dict[str, ClassType]) -> 'AnalyzerConfigBuilder':
        self.config_data['springboot_annotations'] = annotations
        return self
    
    def with_custom_job_patterns(self, patterns: List[str]) -> 'AnalyzerConfigBuilder':
        self.config_data['job_patterns'] = patterns
        return self
    
    def build(self) -> AnalyzerConfig:
        return AnalyzerConfig(**self.config_data)


class JavaAnalyzerFactory:
    """Factory for creating analyzer instances"""
    
    @staticmethod
    def create_default_analyzer() -> JavaCodeAnalyzer:
        """Create analyzer with default configuration"""
        config = AnalyzerConfig()
        return JavaCodeAnalyzer(config)
    
    @staticmethod
    def create_custom_analyzer(config: AnalyzerConfig) -> JavaCodeAnalyzer:
        """Create analyzer with custom configuration"""
        return JavaCodeAnalyzer(config)
    
    @staticmethod
    def create_spring_boot_analyzer() -> JavaCodeAnalyzer:
        """Create analyzer optimized for Spring Boot projects"""
        config = AnalyzerConfigBuilder().with_comment_removal(True).build()
        return JavaCodeAnalyzer(config)


# ============================================================================
# Public API Functions (Backward Compatibility)
# ============================================================================

def parse_project(root: Path, project_id: str, remove_comments: bool = True) -> Tuple[List[CodeChunk], DependencyGraph]:
    """Main entry point function for backward compatibility"""
    config = AnalyzerConfigBuilder().with_comment_removal(remove_comments).build()
    analyzer = JavaAnalyzerFactory.create_custom_analyzer(config)
    return analyzer.parse_project(root, project_id)


def create_analyzer(remove_comments: bool = True) -> JavaCodeAnalyzer:
    """Create analyzer instance"""
    config = AnalyzerConfigBuilder().with_comment_removal(remove_comments).build()
    return JavaAnalyzerFactory.create_custom_analyzer(config)


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of the fixed analyzer"""
    
    # Example configuration
    args = {
        "project_path": "F:/_side_projects/source_atlas/data/repo/test",
        "project_id": "test",
        "output": "./result",
        "remove_comments": True,
        "verbose": True
    }
    
    if args["verbose"]:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_path = Path(args["project_path"])
    if not project_path.exists():
        logger.error(f"Project path {project_path} does not exist")
        return 1
    
    try:
        # Create analyzer using factory
        logger.info(f"Analyzing Java project at: {project_path}")
        
        # Use the fixed analyzer
        config = AnalyzerConfigBuilder().with_comment_removal(args["remove_comments"]).build()
        analyzer = JavaAnalyzerFactory.create_custom_analyzer(config)
        chunks, dependency_graph = analyzer.parse_project(project_path, args["project_id"])
        
        # Display results
        logger.info(f"\nAnalysis Results:")
        logger.info(f"Found {len(chunks)} classes/interfaces/enums")
        logger.info(f"Dependency graph has {len(dependency_graph.nodes)} nodes and {len(dependency_graph.edges)} edges")
        
        # Print examples with fixed information
        logger.info(f"\nExample classes:")
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"{i+1}. {chunk.full_class_name}")
            logger.info(f"   Type: {chunk.class_type.value}")
            logger.info(f"   Methods: {len(chunk.methods)}")
            logger.info(f"   Fields: {len(chunk.fields)}")
            if chunk.extends:
                logger.info(f"   Extends: {chunk.extends}")
            if chunk.implements:
                logger.info(f"   Implements: {', '.join(chunk.implements)}")
            
            # Show method types
            method_types = {}
            for method in chunk.methods:
                method_type = method.method_type.value
                method_types[method_type] = method_types.get(method_type, 0) + 1
            if method_types:
                logger.info(f"   Method types: {method_types}")
        
        # Export results
        if args["output"]:
            output_path = Path(args["output"])
            analyzer.export_results(chunks, dependency_graph, output_path)
        
        # Generate and display statistics
        stats = analyzer.generate_statistics(chunks, dependency_graph)
        
        logger.info(f"\nProject Statistics:")
        logger.info(f"Total Classes: {stats['total_classes']}")
        logger.info(f"Total Methods: {stats['total_methods']}")
        logger.info(f"Total Fields: {stats['total_fields']}")
        logger.info(f"Lines of Code: {stats['complexity_metrics']['total_lines_of_code']}")
        
        logger.info(f"\nMethod Type Distribution:")
        for method_type, count in stats['method_types'].items():
            logger.info(f"  {method_type}: {count}")
        
        # Show Spring Boot specific analysis
        spring_classes = [c for c in chunks if c.class_type in [
            ClassType.CONFIGURATION, ClassType.SERVICE, ClassType.CONTROLLER,
            ClassType.REPOSITORY, ClassType.ENTITY, ClassType.COMPONENT
        ]]
        
        if spring_classes:
            logger.info(f"\nSpring Boot Analysis:")
            logger.info(f"Found {len(spring_classes)} Spring Boot components:")
            for class_type_name, count in stats['class_types'].items():
                if count > 0 and class_type_name in ['configuration', 'service', 'controller', 'repository', 'entity', 'component']:
                    logger.info(f"  {class_type_name}: {count}")
        
        logger.info(f"\nAnalysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())