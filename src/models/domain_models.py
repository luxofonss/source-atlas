from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set

@dataclass
class Field:
    name: str
    type: str
    full_type: str
    annotations: Tuple[str, ...]

@dataclass
class RestEndpoint:
    method: str
    path: str
    produces: str
    consumes: str

@dataclass
class Method:
    name: str
    return_type: str
    full_return_type: str
    parameters: Tuple[Tuple[str, str], ...]
    modifiers: Tuple[str, ...]
    annotations: Tuple[str, ...]
    method_type: 'MethodType'
    body: str
    is_abstract: bool
    is_override: bool
    throws: Tuple[str, ...]
    method_calls: Tuple[str, ...]
    variable_usage: Tuple[str, ...]
    inheritance_info: Tuple[str, ...]
    extends_info: Tuple[str, ...]
    endpoint: Optional[RestEndpoint]

class MethodType(Enum):
    REGULAR = "regular"
    GETTER = "getter"
    SETTER = "setter"
    CONSTRUCTOR = "constructor"
    STATIC = "static"
    REST_ENDPOINT = "rest_endpoint"
    OVERRIDE = "override"

class ClassType(Enum):
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    RECORD = "record"
    ANNOTATION = "annotation"
    ABSTRACT = "abstract"
    CONTROLLER = "controller"
    SERVICE = "service"
    REPOSITORY = "repository"
    CONFIGURATION = "configuration"
    ENTITY = "entity"
    FILTER = "filter"
    CONTROLLER_ADVICE = "controller_advice"
    BEAN = "bean"
    COMPONENT = "component"

@dataclass
class CodeChunk:
    package: str
    class_name: str
    full_class_name: str
    class_type: ClassType
    file_path: str
    content: str
    implements: Tuple[str, ...]
    extends: Optional[str]
    fields: List[Field]
    imports: Dict[str, str]
    methods: List[Method]
    annotations: List[str]
    is_nested: bool
    parent_class: Optional[str]
    inner_classes: Tuple[str, ...]
    modifiers: List[str]

    def to_dict(self) -> Dict:
        """Convert CodeChunk to a JSON-serializable dictionary."""
        return {
            "package": self.package,
            "class_name": self.class_name,
            "full_class_name": self.full_class_name,
            "class_type": self.class_type.value,
            "file_path": self.file_path,
            "content": self.content,
            "implements": list(self.implements),
            "extends": self.extends,
            "fields": [field.__dict__ for field in self.fields],
            "imports": self.imports,
            "methods": [{
                "name": method.name,
                "return_type": method.return_type,
                "full_return_type": method.full_return_type,
                "parameters": list(method.parameters),
                "modifiers": list(method.modifiers),
                "annotations": list(method.annotations),
                "method_type": method.method_type.value,
                "body": method.body,
                "is_abstract": method.is_abstract,
                "is_override": method.is_override,
                "throws": list(method.throws),
                "method_calls": list(method.method_calls),
                "variable_usage": list(method.variable_usage),
                "inheritance_info": list(method.inheritance_info),
                "extends_info": list(method.extends_info),
                "endpoint": method.endpoint.__dict__ if method.endpoint else None
            } for method in self.methods],
            "annotations": self.annotations,
            "is_nested": self.is_nested,
            "parent_class": self.parent_class,
            "inner_classes": list(self.inner_classes),
            "modifiers": self.modifiers
        }

@dataclass
class DependencyGraph:
    nodes: Dict[str, CodeChunk]  # Maps class names to CodeChunk objects
    edges: Set[Tuple[str, str, str]]  # (source, target, relationship)

    def __init__(self):
        self.nodes = {}
        self.edges = set()
    
    def add_node(self, class_name: str, chunk: CodeChunk) -> None:
        """Add a node to the graph."""
        self.nodes[class_name] = chunk
    
    def add_edge(self, source: str, target: str, relationship: str) -> None:
        """Add an edge to the graph."""
        self.edges.add((source, target, relationship))
    
    def to_dict(self) -> Dict:
        """Convert DependencyGraph to a JSON-serializable dictionary."""
        return {
            "nodes": {name: chunk.to_dict() for name, chunk in self.nodes.items()},
            "edges": [{"source": source, "target": target, "relationship": rel} 
                      for source, target, rel in self.edges]
        }