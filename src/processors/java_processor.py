import logging
import re
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple

from tree_sitter import Language, Parser, Node

from processors.base_processor import BaseFileProcessor
from models.domain_models import CodeChunk, ClassType, Field, Method, MethodType, RestEndpoint
from models.analyzer_config import AnalyzerConfig
from services.comment_remover import JavaCommentRemover
from services.type_resolver import JavaTypeResolver
from services.spring_boot_method_type_detector import SpringBootMethodTypeDetector
from services.rest_endpoint_extractor import RestEndpointExtractor

logger = logging.getLogger(__name__)

class JavaFileProcessor(BaseFileProcessor):
    """Processor for Java source files."""
    
    def __init__(self, config: AnalyzerConfig, language: Language, parser: Parser):
        super().__init__(config, language, parser)
        self.comment_remover = JavaCommentRemover()
        self.method_type_detector = SpringBootMethodTypeDetector(config)
        self.endpoint_extractor = RestEndpointExtractor()
    
    def process_file(self, file_path: Path, project_id: str, class_cache: Set[str] = None) -> List[CodeChunk]:
        """Process a single Java file and return chunks for all classes/interfaces."""
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
            
            type_resolver = JavaTypeResolver(self.config, class_cache or set())
            all_class_nodes = self._extract_all_class_nodes(root_node)
            chunks = []
            
            for class_node in all_class_nodes:
                chunk = self._parse_class_node(
                    class_node, content, package, imports, str(file_path), root_node, type_resolver
                )
                if chunk:
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding fallback."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin1') as f:
                return f.read()
    
    def _extract_package(self, root_node: Node, content: str) -> str:
        """Extract package declaration."""
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
        """Extract import statements."""
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
        """Extract all class nodes including nested ones."""
        all_class_nodes = []
        try:
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
        """Parse a single class node."""
        try:
            class_name = self._get_class_name(class_node, content)
            if not class_name:
                return None
            
            is_nested = self._is_nested_class(class_node, root_node)
            full_class_name = self._build_full_class_name(class_name, package, class_node, content, root_node)
            
            modifiers = self._extract_modifiers(class_node, content)
            annotations = self._extract_annotations(class_node, content)
            class_type = self._detect_class_type(class_node, annotations, modifiers)
            
            implements = self._extract_implements(class_node, content, imports, package, type_resolver)
            extends = self._extract_extends(class_node, content, imports, package, type_resolver)
            fields = self._extract_class_fields(class_node, content, imports, package, type_resolver)
            methods = self._extract_class_methods(class_node, content, imports, package, 
                                                implements, extends, class_name, full_class_name, 
                                                type_resolver, fields)
            
            return CodeChunk(
                package=package,
                class_name=class_name,
                full_class_name=full_class_name,
                class_type=class_type,
                file_path=file_path,
                content=content[class_node.start_byte:class_node.end_byte],
                implements=implements,
                extends=extends,
                fields=fields,
                imports=imports,
                methods=methods,
                annotations=annotations,
                is_nested=is_nested,
                parent_class=self._get_parent_class(class_node, content, package) if is_nested else None,
                inner_classes=self._get_direct_inner_classes(class_node, content, class_name, package),
                modifiers=modifiers
            )
        except Exception as e:
            logger.error(f"Error parsing class node: {e}")
            return None
    
    def _get_class_name(self, class_node: Node, content: str) -> Optional[str]:
        """Extract class name from class node."""
        try:
            for child in class_node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
            return None
        except Exception:
            return None
    
    def _extract_modifiers(self, class_node: Node, content: str) -> List[str]:
        """Extract modifiers from class node."""
        modifiers = []
        try:
            for child in class_node.children:
                if child.type == 'modifiers':
                    for modifier in child.children:
                        if modifier.type in ['public', 'private', 'protected', 'static', 
                                            'final', 'abstract', 'synchronized', 'volatile', 
                                            'transient', 'native']:
                            modifiers.append(content[modifier.start_byte:modifier.end_byte])
        except Exception as e:
            logger.debug(f"Error extracting modifiers: {e}")
        return modifiers
    
    def _extract_annotations(self, class_node: Node, content: str) -> List[str]:
        """Extract annotations from class node."""
        annotations = []
        try:
            for child in class_node.children:
                if child.type == 'modifiers':
                    for modifier in child.children:
                        if modifier.type == 'annotation':
                            annotation_text = content[modifier.start_byte:modifier.end_byte]
                            annotations.append(annotation_text)
        except Exception as e:
            logger.debug(f"Error extracting annotations: {e}")
        return annotations
    
    def _is_nested_class(self, class_node: Node, root_node: Node) -> bool:
        """Check if class is nested."""
        parent = class_node.parent
        while parent and parent != root_node:
            if parent.type in ['class_declaration', 'interface_declaration', 
                              'enum_declaration', 'record_declaration']:
                return True
            parent = parent.parent
        return False
    
    def _build_full_class_name(self, class_name: str, package: str, class_node: Node, 
                              content: str, root_node: Node) -> str:
        """Build fully qualified class name, including nested classes."""
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
            parent_names.reverse()
            nested_path = '.'.join(parent_names + [class_name])
            return f"{package}.{nested_path}" if package else nested_path
        return f"{package}.{class_name}" if package else class_name
    
    def _get_parent_class(self, class_node: Node, content: str, package: str) -> Optional[str]:
        """Get the fully qualified name of the parent class for nested classes."""
        parent = class_node.parent
        while parent:
            if parent.type in ['class_declaration', 'interface_declaration', 
                              'enum_declaration', 'record_declaration']:
                parent_name = self._get_class_name(parent, content)
                if parent_name:
                    parent_full_name = self._build_full_class_name(parent_name, package, parent, content, parent)
                    return parent_full_name
            parent = parent.parent
        return None
    
    def _get_direct_inner_classes(self, class_node: Node, content: str, class_name: str, package: str) -> Tuple[str, ...]:
        """Extract fully qualified names of direct inner classes."""
        inner_classes = []
        try:
            for child in class_node.children:
                if child.type == 'class_body':
                    for body_child in child.children:
                        if body_child.type in ['class_declaration', 'interface_declaration', 
                                              'enum_declaration', 'record_declaration']:
                            inner_class_name = self._get_class_name(body_child, content)
                            if inner_class_name:
                                full_inner_class_name = f"{package}.{class_name}.{inner_class_name}" if package else f"{class_name}.{inner_class_name}"
                                inner_classes.append(full_inner_class_name)
        except Exception as e:
            logger.debug(f"Error extracting inner classes: {e}")
        return tuple(inner_classes)
    
    def _detect_class_type(self, class_node: Node, annotations: List[str], modifiers: List[str]) -> ClassType:
        """Detect class type based on annotations and modifiers."""
        class_type = class_node.type
        if class_type == 'class_declaration':
            if 'abstract' in modifiers:
                return ClassType.ABSTRACT
            return ClassType.CLASS
        elif class_type == 'interface_declaration':
            return ClassType.INTERFACE
        elif class_type == 'enum_declaration':
            return ClassType.ENUM
        elif class_type == 'record_declaration':
            return ClassType.RECORD
        elif class_type == 'annotation_type_declaration':
            return ClassType.ANNOTATION
        
        for annotation in annotations:
            for spring_anno, c_type in self.config.springboot_annotations.items():
                if spring_anno in annotation:
                    return c_type
        return ClassType.CLASS
    
    def _extract_implements(self, class_node: Node, content: str, 
                          imports: Dict[str, str], package: str, type_resolver: JavaTypeResolver) -> Tuple[str, ...]:
        """Extract interfaces implemented by the class."""
        interfaces = []
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
                interface_node = capture[0]
                interface_name = content[interface_node.start_byte:interface_node.end_byte]
                resolved_types = type_resolver.resolve_type_name(interface_name, imports, package)
                interfaces.extend(resolved_types)
        except Exception as e:
            logger.debug(f"Error extracting implements: {e}")
        return tuple(interfaces)
    
    def _extract_extends(self, class_node: Node, content: str, 
                        imports: Dict[str, str], package: str, type_resolver: JavaTypeResolver) -> Optional[str]:
        """Extract extended class."""
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
        """Extract ONLY fields that belong directly to this class."""
        fields = []
        try:
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
        """Parse a single field declaration."""
        try:
            field_type = None
            field_name = None
            field_annotations = []
            
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
                    annotations=tuple(field_annotations)
                )
        except Exception as e:
            logger.debug(f"Error parsing field declaration: {e}")
        return None
    
    def _extract_class_methods(self, class_node: Node, content: str, imports: Dict[str, str],
                              package: str, implements: List[str], extends: Optional[str],
                              class_name: str, full_class_name: str, type_resolver: JavaTypeResolver,
                              fields: List[Field]) -> List[Method]:
        """Extract ONLY methods that belong directly to this class."""
        methods = []
        try:
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
        """Process a single method or constructor node."""
        try:
            method_name = None
            for child in method_node.children:
                if child.type == 'identifier':
                    method_name = content[child.start_byte:child.end_byte]
                    break
            if not method_name:
                return None
            
            modifiers = self._extract_modifiers(method_node, content)
            annotations = self._extract_annotations(method_node, content)
            return_type, full_return_type = self._extract_return_type(method_node, content, type_resolver, imports, package) if not is_constructor else ("", "")
            parameters = self._extract_method_parameters(method_node, content, type_resolver, imports, package)
            throws = self._extract_throws(method_node, content, type_resolver, imports, package)
            
            body = ""
            method_calls = []
            variable_usage = []
            for child in method_node.children:
                if child.type == 'block':
                    body = content[method_node.start_byte:method_node.end_byte]
                    method_calls, variable_usage = self._extract_method_calls_and_usage(
                        child, content, imports, package, full_class_name, type_resolver, parameters, fields
                    )
                    break
            
            is_abstract = 'abstract' in modifiers and not body.strip()
            is_override = '@Override' in annotations
            method_type = self.method_type_detector.detect_method_type(
                method_name, is_constructor, body, annotations, return_type, parameters
            )
            endpoint = self.endpoint_extractor.extract_from_method(method_node, content, class_node)
            inheritance_info = self._get_correct_inheritance_info(method_name, implements, extends, parameters, type_resolver)
            extends_info = self._get_correct_extends_info(method_name, extends, parameters, type_resolver)
            
            return Method(
                name=f"{full_class_name}.{method_name}",
                return_type=return_type,
                full_return_type=full_return_type,
                parameters=tuple(parameters),
                modifiers=tuple(modifiers),
                annotations=tuple(annotations),
                method_type=method_type,
                body=body,
                is_abstract=is_abstract,
                is_override=is_override,
                throws=tuple(throws),
                method_calls=tuple(method_calls),
                variable_usage=tuple(variable_usage),
                inheritance_info=tuple(inheritance_info),
                extends_info=tuple(extends_info),
                endpoint=endpoint
            )
        except Exception as e:
            logger.debug(f"Error processing method node: {e}")
            return None
    
    def _extract_return_type(self, method_node: Node, content: str, type_resolver: JavaTypeResolver,
                            imports: Dict[str, str], package: str) -> Tuple[str, str]:
        """Extract return type from method."""
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
        """Extract method parameters as (name, type) tuples."""
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
        """Extract throws clause."""
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
        """Extract method calls and variable usage from method body."""
        method_calls = set()
        variable_usage = set()
        try:
            local_variables = self._extract_local_variables(body_node, content, type_resolver, imports, package)
            for _, param_type in parameters:
                clean_type = self._extract_base_class_name(param_type)
                if clean_type and not self._is_builtin_type(clean_type):
                    variable_usage.add(clean_type)
            for field in fields:
                clean_type = self._extract_base_class_name(field.full_type)
                if clean_type and not self._is_builtin_type(clean_type):
                    variable_usage.add(clean_type)
            for var_type in local_variables.values():
                clean_type = self._extract_base_class_name(var_type)
                if clean_type and not self._is_builtin_type(clean_type):
                    variable_usage.add(clean_type)
            
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
        """Extract local variable declarations and their types."""
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
        """Extract base class name from type, removing generics and arrays."""
        if not type_name:
            return ""
        clean_type = type_name.replace('[]', '')
        generic_start = clean_type.find('<')
        if generic_start != -1:
            clean_type = clean_type[:generic_start]
        return clean_type.strip()
    
    def _is_builtin_type(self, type_name: str) -> bool:
        """Check if type is a built-in Java type."""
        base_type = type_name.split('.')[0] if '.' in type_name else type_name
        base_type = re.sub(r'<.*?>', '', base_type).replace('[]', '')
        return base_type in self.config.builtin_types or base_type.startswith('java.lang.')
    
    def _is_class_reference(self, name: str, imports: Dict[str, str], package: str) -> bool:
        """Check if name refers to a class."""
        return name in imports or (name and name[0].isupper())
    
    def _get_correct_inheritance_info(self, method_name: str, implements: List[str], 
                                    extends: Optional[str], parameters: List[Tuple[str, str]], 
                                    type_resolver: JavaTypeResolver) -> List[str]:
        """Get inheritance information for methods."""
        inheritance_sources = []
        param_types = [ptype for _, ptype in parameters]
        param_sig = ','.join(param_types)
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
        """Get extends information for methods."""
        if not extends:
            return []
        known_methods = {'equals', 'hashCode', 'toString'}
        if method_name in known_methods:
            param_types = [ptype for _, ptype in parameters]
            param_sig = ','.join(param_types)
            return [f"{extends}.{method_name}({param_sig})"]
        return []