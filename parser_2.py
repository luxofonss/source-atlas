from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from networkx.algorithms import hierarchy
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser

from parser import DependencyGraph
from utils.constant import JAVA_STANDARD_TYPES, GENERIC_TYPE_VARS

JAVA_LANGUAGE: Language = get_language("java")
_PARSER = Parser()
_PARSER.set_language(JAVA_LANGUAGE)
EXCLUDE_CHUNK_TYPE = ["package_declaration", "import_declaration"]

BLACKLIST_DIR = {
    ".git", ".idea", "env", ".github", ".gitlab", "target", "build", 
    "out", "bin", ".vscode", "node_modules", "__pycache__"
}
WHITELIST_EXT = {".java"}

# Consolidated patterns for better maintainability
PATTERNS = {
    "job_annotations": {
        "@Scheduled", "@Async", "@EventListener", "@JmsListener", "@KafkaListener",
        "@RabbitListener", "@Job", "@Step", "@StepScope", "@JobScope",
        "@DisallowConcurrentExecution", "@PersistJobDataAfterExecution",
        "@Retryable", "@BackgroundJob", "@Cron", "@FixedRate", "@FixedDelay"
    },
    "event_publishers": {
        "applicationEventPublisher.publishEvent", "eventPublisher.publishEvent", 
        "publish", "send", "emit", "fire", "trigger"
    },
    "event_listeners": {
        "@EventListener", "@JmsListener", "@KafkaListener", "@RabbitListener", 
        "@Subscribe", "@Handler"
    },
    "schedulers": {
        "@Scheduled", "TaskScheduler", "ScheduledExecutorService", "@EnableScheduling"
    }
}

@dataclass
class ClassNode:
     extends: Optional[str] = None
     implements: List[str] = None
     methods: List[Dict[str, str]] = None
     file_path: str = ""

     def __post_init__(self):
          if self.implements is None:
               self.implements = []
          if self.methods is None:
               self.methods = []

@dataclass
class ClassHierarchy:
    extends: Optional[str] = None
    implements: List[str] = None
    methods: List[Dict[str, str]] = None
    file_path: str = ""
    
    def __post_init__(self):
        if self.implements is None:
            self.implements = []
        if self.methods is None:
            self.methods = []

CodeChunk = Dict[str, object]
DependencyGraph = Dict[str, Dict[str, List[str]]]

class JavaCodeAnalyzer:
     def __init__(self, remove_comments: bool = True):
          self.remove_comments = remove_comments
          self.class_hierarchy: Dict[str, ClassHierarchy] = {}
          self.event_registry: Dict[str, List[str]] = {}

     def parse_project(self, root: Path, project_id: str) -> Tuple[List[CodeChunk], DependencyGraph]:
          chunks: List[CodeChunk] = []
          dep_graph: DependencyGraph = {}

          java_files = self._list_java_files(root)

          for file_path in java_files:
               try:
                    source = self._read_file(file_path)
                    tree = _PARSER.parse(source.encode("utf-8"))
                    class_nodes = self._get_class_nodes(tree.root_node)
                    for class_node in class_nodes:
                         class_name = self._get_identifier(class_node)
                         implements = []
                         extends = None

                         superclass_node = class_node.child_by_field_name("superclass")
                         if superclass_node:
                              type_id_node = (superclass_node.child_by_field_name("type_identifier")) or next((c for c in superclass_node.children if c.type == "type_identifier"), None)
                              if type_id_node:
                                   extends = source[type_id_node.start_byte:type_id_node.end_byte]
                         interfaces_node = class_node.child_by_field_name("interfaces")
                         if interfaces_node:
                              type_list = next((c for c in interfaces_node.children if c.type == "type_list"), None)
                              if type_list:
                                   for child in type_list.children:
                                        if child.type == "type_identifier":
                                             implements.append(source[child.start_byte:child.end_byte])
                         hierarchy = {"extends": extends, "implements": implements}

                         methods = self._extract_method_signatures(class_node, source)
                         
                         self.class_hierarchy[class_name] = ClassHierarchy(
                              extends=hierarchy["extends"],
                              implements=hierarchy["implements"],
                              methods=methods,
                              file_path=str(file_path)
                         )

                    file_chunks, file_graph = self._parse_file(project_id, file_path)
                    chunks.extend(file_chunks)
                    dep_graph.update(file_graph)
               except Exception as exc:
                    print("err")
                    raise

     def _parse_file(self, project_id: str, file_path: Path) -> Tuple[List[CodeChunk], DependencyGraph]:
          """Parse a single Java file"""
          try:
               source = self._read_file(file_path)
               tree = _PARSER.parse(source.encode("utf-8"))
               return self._extract_chunks_from_tree(project_id, file_path, tree, source)
          except Exception as exc:
               return [], {}
     
     def _extract_chunks_from_tree(self, project_id: str, file_path: Path, tree, source: str) -> Tuple[List[CodeChunk], DependencyGraph]:
          """Extract chunks and dependency graph from parsed tree"""
          chunks = []
          graph = {}
          
          for class_node in [n for n in tree.root_node.children if n.type not in EXCLUDE_CHUNK_TYPE]:
               class_chunks, class_graph = self._process_class_node(project_id, file_path, class_node, source)
               chunks.extend(class_chunks)
               graph.update(class_graph)
          
          return chunks, graph

     def _process_class_node(self, project_id: str, file_path: Path, class_node, source: str) -> Tuple[List[CodeChunk], DependencyGraph]:
          """Process a single class node"""
          chunks = []
          graph = {}
          
          class_name = self._get_identifier(class_node, source) or file_path.stem
          chunk_type = self._infer_chunk_type(class_node, source)
          class_endpoints = self._extract_class_level_endpoints(class_node, source)
          hierarchy_info = self._extract_class_hierarchy(class_node, source)
          field_map = self._extract_fields(class_node, source)
          
          # Create class-level chunk
          class_chunk = self._build_chunk(
               project_id=project_id, file_path=file_path, class_name=class_name,
               method_name=None, chunk_type=chunk_type, node=class_node, source=source,
               calls=[], endpoints=self._format_endpoints(class_endpoints, "REQUEST"),
               extends=hierarchy_info['extends'], implements=hierarchy_info['implements']
          )
          chunks.append(class_chunk)
          graph[class_name] = self._create_graph_node(hierarchy_info)
          
          # Process methods
          method_nodes = self._get_method_nodes(class_node)
          for method_node in method_nodes:
               method_chunk, method_graph_entry = self._process_method_node(
                    project_id, file_path, class_name, method_node, source, field_map, chunk_type
               )
               chunks.append(method_chunk)
               graph.update(method_graph_entry)
          
          return chunks, graph
     
     def _process_method_node(self, project_id: str, file_path: Path, class_name: str, 
                              method_node, source: str, field_map: Dict[str, str], 
                              chunk_type: str) -> Tuple[CodeChunk, Dict]:
          """Process a single method node"""
          method_name = self._get_identifier(method_node, source) or "unknown"
          method_signature = self._extract_method_signature(method_node, source)
          param_map = self._extract_param_types(method_node, source)
          vars_list = self._filter_vars(self._extract_vars(method_node))
          calls = self._extract_calls(method_node, source, class_name, {**field_map, **param_map})
          
          # Method inheritance and endpoints
          method_inheritance = self._resolve_method_inheritance(class_name, method_name, method_signature)
          endpoint = self._extract_method_endpoint(method_node, source)
          endpoints = [{"path": endpoint[0], "method": endpoint[1]}] if endpoint else []
          
          # Job information
          is_job = self._is_job_method(method_node, source)
          job_info = self._extract_job_info(method_node, source, class_name) if is_job else None
          
          chunk = self._build_chunk(
               project_id=project_id, file_path=file_path, class_name=class_name,
               method_name=method_name, chunk_type=chunk_type, node=method_node, source=source,
               calls=calls, endpoints=endpoints, extends=method_inheritance["extends"],
               implements=method_inheritance["implements"], vars=vars_list,
               is_job=is_job, job_info=job_info
          )
          
          graph_entry = {f"{class_name}.{method_name}": {"calls": calls, "called_by": []}}
          
          return chunk, graph_entry

     def _extract_class_level_endpoints(self, class_node, source: str) -> List[str]:
          """Extract endpoints from class-level RequestMapping annotations"""
          paths = []
          for child in class_node.children:
               if child.type == "modifiers":
                    text = source[child.start_byte:child.end_byte]
                    if "@RequestMapping" in text:
                         val = (self._extract_annotation_value(text, "value") or 
                              self._extract_annotation_value(text, "path"))
                         if val:
                              paths.append(val)
          return paths
          
     def _infer_chunk_type(self, node, source: str) -> str:
          """Infer the type of a code chunk based on annotations"""
          annotations = [c for c in node.children if c.type == "modifiers"]
          ann_text = "".join(source[c.start_byte:c.end_byte].lower() for c in annotations)
          
          # Job-related types (check first for priority)
          job_type_mapping = {
               "@scheduled": "scheduled_job",
               "@eventlistener": "event_listener", 
               "@async": "async_job",
               "@jmslistener": "message_listener",
               "@kafkalistener": "message_listener",
               "@rabbitlistener": "message_listener",
               "@job": "batch_job",
               "@step": "batch_job"
          }
          
          for annotation, job_type in job_type_mapping.items():
               if annotation in ann_text:
                    return job_type
          
          if any(job_ann.lower() in ann_text for job_ann in PATTERNS["job_annotations"]):
               return "job"
          
          # Standard Spring types
          spring_type_mapping = {
               "@controller": "controller",
               "@restcontroller": "controller",
               "@service": "service",
               "@repository": "repository",
               "@entity": "entity",
               "filter": "filter",
               "component": "component",
               "bean": "bean",
               "abstract": "abstract_class",
               "configuration": "configuration",
               "controlleradvice": "configuration"
          }
          
          for annotation, spring_type in spring_type_mapping.items():
               if annotation in ann_text:
                    return spring_type
          
          return "interface" if node.type == "interface_declaration" else "other"
     
     
     def _extract_method_signatures(self, class_node, source: str) -> List[Dict[str, str]]:
          """Extract method signatures from a class"""
          method_signatures = []
          method_nodes = self._get_method_nodes(class_node)
          
          for method_node in method_nodes:
               method_name = self._get_identifier(method_node, source) or "unknown"
               signature = self._extract_method_signature(method_node, source)
               method_signatures.append({"name": method_name, "signature": signature})
          
          return method_signatures

     def _get_identifier(self, node, source: str) -> Optional[str]:
          for child in node.children:
               if child.type == "identifier":
                    return source[child.start_byte:child.end_byte]
          return None

     def _get_class_nodes(self, root_node, parent_class: str = None) -> List:
          result = []
          if root_node.type == "class_declaration":
               result.append(root_node)
          for child in root_node.children:
               result.extend(self._get_class_nodes(child, parent_class))
          return result
               

     def _list_java_files(self, root: Path) -> List[Path]:
          files = []
          for p in root.rglob("*.java"):
               if p.suffix in WHITELIST_EXT and not self._should_skip(p):
                    files.append(p)
          return files

     def _should_skip(self, path: Path) -> bool:
          parts = {p.name for p in path.parents}
          return bool(parts & BLACKLIST_DIR)

     def _read_file(self, file_path: Path) -> str:
          try:
               text = file_path.read_text("utf-8")
               return self._remove_comments(text)
          except UnicodeDecodeError:
               raise

     def _remove_comments(self, source: str) -> str:
          """Remove all Java comments from source code"""
          # Remove single-line comments
          source = re.sub(r'//.*?$', '', source, flags=re.MULTILINE)
          # Remove multi-line and Javadoc comments
          source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
          
          # Clean up whitespace
          lines = [line for line in source.split('\n') if line.strip()]
          return '\n'.join(lines)
    

def parse_project(root: Path, project_id: str, remove_comments: bool = True) -> Tuple[List[CodeChunk], DependencyGraph]:
    """Main entry point function for backward compatibility"""
    analyzer = JavaCodeAnalyzer(remove_comments)
    return analyzer.parse_project(root, project_id)