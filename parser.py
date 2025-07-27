from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass, asdict

from loguru import logger
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from utils.constant import JAVA_STANDARD_TYPES, GENERIC_TYPE_VARS
from pyvis.network import Network

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
class JobInfo:
    type: str = "unknown"
    schedule: Optional[Dict] = None
    async_job: bool = False
    event_types: List[str] = None
    topics: List[str] = None
    retry_config: Dict = None
    class_name: str = ""
    method_name: str = ""
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = []
        if self.topics is None:
            self.topics = []
        if self.retry_config is None:
            self.retry_config = {}

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
        self.job_registry: Dict[str, JobInfo] = {}
        
    def parse_project(self, root: Path, project_id: str) -> Tuple[List[CodeChunk], DependencyGraph]:
        """Main entry point for parsing a Java project"""
        chunks: List[CodeChunk] = []
        dep_graph: DependencyGraph = {}
        
        java_files = self._list_java_files(root)
        
        # First pass: collect metadata
        logger.info("First pass: collecting class hierarchy and job/event information")
        for file_path in java_files:
            self._collect_file_metadata(file_path)
        
        # Second pass: parse files with context
        logger.info("Second pass: parsing files with context")
        for i, file_path in enumerate(java_files):
            logger.info(f"Processing file {i+1}/{len(java_files)}: {file_path}")
            file_chunks, file_graph = self._parse_file(project_id, file_path)
            chunks.extend(file_chunks)
            dep_graph.update(file_graph)
        
        # Post-processing
        chunks = self._remove_duplicates(chunks)
        self._build_relationships(chunks, dep_graph)
        
        logger.info(f"Parse complete: {len(chunks)} chunks, {len(dep_graph)} graph nodes")
        logger.info(f"Event registry: {len(self.event_registry)} event types")
        logger.info(f"Job registry: {len(self.job_registry)} jobs")
        
        return chunks, dep_graph
    
    def _list_java_files(self, root: Path) -> List[Path]:
        """List all Java files in the project, excluding blacklisted directories"""
        files = []
        for p in root.rglob("*.java"):
            if p.suffix in WHITELIST_EXT and not self._should_skip(p):
                files.append(p)
        return files
    
    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped based on blacklist"""
        parts = {p.name for p in path.parents}
        return bool(parts & BLACKLIST_DIR)
    
    def _collect_file_metadata(self, file_path: Path):
        """First pass: collect class hierarchy and job/event information"""
        try:
            source = self._read_file(file_path)
            tree = _PARSER.parse(source.encode("utf-8"))
            
            self._collect_class_hierarchy(file_path, tree, source)
            self._collect_events_and_jobs(file_path, tree, source)
        except Exception as exc:
            logger.error(f"Error collecting metadata for {file_path}: {exc}")
    
    def _read_file(self, file_path: Path) -> str:
        """Read and optionally clean file content"""
        try:
            text = file_path.read_text("utf-8")
            return self._remove_comments_from_source(text) if self.remove_comments else text
        except UnicodeDecodeError:
            logger.warning(f"Could not read {file_path} – skipping")
            raise
    
    def _parse_file(self, project_id: str, file_path: Path) -> Tuple[List[CodeChunk], DependencyGraph]:
        """Parse a single Java file"""
        try:
            source = self._read_file(file_path)
            tree = _PARSER.parse(source.encode("utf-8"))
            return self._extract_chunks_from_tree(project_id, file_path, tree, source)
        except Exception as exc:
            logger.error(f"Parser error for {file_path}: {exc}")
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
    
    def _collect_class_hierarchy(self, file_path: Path, tree, source: str):
        """Collect class hierarchy information"""
        for class_node in self._get_class_nodes(tree.root_node):
            class_name = self._get_identifier(class_node, source) or file_path.stem
            hierarchy = self._extract_class_hierarchy(class_node, source)
            methods = self._extract_method_signatures(class_node, source)
            
            self.class_hierarchy[class_name] = ClassHierarchy(
                extends=hierarchy["extends"],
                implements=hierarchy["implements"],
                methods=methods,
                file_path=str(file_path)
            )
    
    def _collect_events_and_jobs(self, file_path: Path, tree, source: str):
        """Collect event publishers/listeners and job information"""
        for method_node in self._get_all_methods(tree.root_node):
            self._process_method_for_events_and_jobs(method_node, source)
    
    def _process_method_for_events_and_jobs(self, method_node, source: str):
        """Process a method for event and job information"""
        method_content = source[method_node.start_byte:method_node.end_byte]
        method_name = self._get_identifier(method_node, source) or "unknown"
        parent_class = self._find_parent_class(method_node, source)
        full_method_name = f"{parent_class}.{method_name}"
        
        # Event processing
        self._process_event_publishers(method_content, full_method_name)
        self._process_event_listeners(method_node, source, full_method_name, method_content)
        
        # Job processing
        if self._is_job_method(method_node, source):
            job_info = self._extract_job_info(method_node, source, parent_class)
            self.job_registry[full_method_name] = job_info
            logger.info(f"Found job: {full_method_name} -> {asdict(job_info)}")
    
    def _build_relationships(self, chunks: List[CodeChunk], dep_graph: DependencyGraph):
        """Build all relationships between chunks"""
        self._populate_called_by(dep_graph)
        self._attach_called_by_to_chunks(chunks, dep_graph)
        self._populate_extends_and_implements_by(chunks, dep_graph)
        self._link_event_publishers_and_listeners(chunks, dep_graph)
        self._link_job_dependencies(chunks, dep_graph)
    
    # Utility methods (consolidated and optimized)
    
    def _get_identifier(self, node, source: str) -> Optional[str]:
        """Get identifier from a node"""
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte]
        
        cursor = node.walk()
        while cursor.goto_next_sibling():
            if cursor.node.type == "identifier":
                return source[cursor.node.start_byte:cursor.node.end_byte]
        return None
    
    def _get_class_nodes(self, root_node, parent_class: str = None) -> List:
        """Get all class nodes from root"""
        source = root_node.source
        result = []
        if root_node.type == "class_declaration":
            if parent_class is None:
                parent_class = self._get_identifier(root_node, source) or "unknown"
            result.append(root_node)
        for child in root_node.children:
            result.extend(self._get_class_nodes(child, parent_class))
        return result
    
    def _get_all_methods(self, root_node) -> List:
        """Get all method nodes from root"""
        result = []
        if root_node.type == "method_declaration":
            result.append(root_node)
        for child in root_node.children:
            result.extend(self._get_all_methods(child))
        return result
    
    def _get_method_nodes(self, class_node) -> List:
        """Get method nodes from a class"""
        method_nodes = []
        for child in class_node.children:
            if child.type in ["class_body", "interface_body", "enum_body"]:
                method_nodes.extend([
                    n for n in child.children if n.type == "method_declaration"
                ])
        return method_nodes
    
    def _filter_vars(self, vars_list: List[str]) -> List[str]:
        """Filter out standard types and generic variables"""
        return [var for var in vars_list 
                if var not in JAVA_STANDARD_TYPES and var not in GENERIC_TYPE_VARS]
    
    def _format_endpoints(self, paths: List[str], method: str) -> List[Dict[str, str]]:
        """Format endpoint paths"""
        return [{"path": path, "method": method} for path in paths]
    
    def _create_graph_node(self, hierarchy_info: Dict) -> Dict:
        """Create a graph node with hierarchy information"""
        return {
            "calls": [], "called_by": [], 
            "extends": hierarchy_info.get('extends'), 
            "implements": hierarchy_info.get('implements', []),
            "extended_by": [], "implemented_by": []
        }
    
    def _find_parent_class(self, method_node, source: str) -> str:
        """Find the parent class of a method"""
        parent = method_node.parent
        while parent:
            if parent.type == "class_declaration":
                return self._get_identifier(parent, source) or "unknown"
            parent = parent.parent
        return "unknown"
    
    def _remove_duplicates(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Remove duplicate chunks based on their ID"""
        seen_ids = set()
        unique_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            else:
                duplicate_count += 1
                logger.debug(f"Duplicate chunk found with ID: {chunk_id}")
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate chunks")
        
        return unique_chunks
    
    # Event and Job Processing Methods
    def _process_event_publishers(self, method_content: str, full_method_name: str):
        """Process event publishers in method content"""
        for pattern in PATTERNS["event_publishers"]:
            if pattern in method_content:
                event_type = self._extract_event_type_from_publisher(method_content, pattern)
                if event_type:
                    self.event_registry.setdefault(event_type, []).append(full_method_name)
                    logger.info(f"Found event publisher: {full_method_name} -> {event_type}")
    
    def _process_event_listeners(self, method_node, source: str, full_method_name: str, method_content: str):
        """Process event listeners in method"""
        annotations = self._extract_method_annotations(method_node, source)
        for annotation in annotations:
            if any(listener_pattern in annotation for listener_pattern in PATTERNS["event_listeners"]):
                event_type = self._extract_event_type_from_listener(annotation, method_content)
                if event_type:
                    self.event_registry.setdefault(event_type, []).append(full_method_name)
                    logger.info(f"Found event listener: {full_method_name} -> {event_type}")
    
    def _extract_event_type_from_publisher(self, method_content: str, pattern: str) -> Optional[str]:
        """Extract event type from publisher method"""
        if "publishEvent" in pattern:
            event_match = re.search(r'publishEvent\s*\(\s*(?:new\s+(\w+)|(\w+))', method_content)
            if event_match:
                return event_match.group(1) or event_match.group(2)
        
        if pattern in ["send", "publish", "emit"]:
            topic_match = re.search(rf'{pattern}\s*\([^,]+,\s*"([^"]+)"', method_content)
            if topic_match:
                return topic_match.group(1)
        
        return None
    
    def _extract_event_type_from_listener(self, annotation: str, method_content: str) -> Optional[str]:
        """Extract event type from listener annotation or method signature"""
        # From @EventListener(CustomEvent.class)
        event_match = re.search(r'@EventListener\s*\(\s*(\w+)\.class\s*\)', annotation)
        if event_match:
            return event_match.group(1)
        
        # From method parameter type
        param_match = re.search(r'public\s+void\s+\w+\s*\(\s*(\w+)\s+\w+\s*\)', method_content)
        if param_match and "Event" in param_match.group(1):
            return param_match.group(1)
        
        # From Kafka/JMS listener topics
        topic_match = re.search(r'topics?\s*=\s*[{"\']*([^}"\',]+)[}"\']*', annotation)
        if topic_match:
            return topic_match.group(1)
        
        return None
    
    def _is_job_method(self, method_node, source: str) -> bool:
        """Check if method is a job"""
        annotations = self._extract_method_annotations(method_node, source)
        annotation_text = " ".join(annotations).lower()
        return any(job_ann.lower() in annotation_text for job_ann in PATTERNS["job_annotations"])
    
    def _extract_job_info(self, method_node, source: str, class_name: str) -> JobInfo:
        """Extract job configuration information"""
        annotations = self._extract_method_annotations(method_node, source)
        method_name = self._get_identifier(method_node, source) or "unknown"
        
        job_info = JobInfo(class_name=class_name, method_name=method_name)
        
        for annotation in annotations:
            self._process_job_annotation(annotation, job_info, source, method_node)
        
        return job_info
    
    def _process_job_annotation(self, annotation: str, job_info: JobInfo, source: str, method_node):
        """Process a single job annotation"""
        annotation_lower = annotation.lower()
        
        if "@scheduled" in annotation_lower:
            job_info.type = "scheduled"
            job_info.schedule = self._extract_schedule_config(annotation)
        elif "@async" in annotation_lower:
            job_info.async_job = True
            if job_info.type == "unknown":
                job_info.type = "async"
        elif "@eventlistener" in annotation_lower:
            job_info.type = "event_listener"
            event_type = self._extract_event_type_from_listener(
                annotation, source[method_node.start_byte:method_node.end_byte]
            )
            if event_type:
                job_info.event_types.append(event_type)
        elif any(listener in annotation_lower for listener in ["@jmslistener", "@kafkalistener", "@rabbitlistener"]):
            job_info.type = "message_listener"
            job_info.topics.extend(self._extract_topics_from_annotation(annotation))
        elif "@retryable" in annotation_lower:
            job_info.retry_config = self._extract_retry_config(annotation)
    
    # Remaining methods (extract_calls, build_chunk, etc.) with optimizations...
    # [Implementation continues with remaining methods, optimized for clarity and performance]
    
    def _extract_calls(self, method_node, source: str, this_class: str, var_types: Dict[str, str]) -> List[str]:
        """Extract method calls from a method node"""
        calls = set()
        
        def walk(node):
            if node.type == "method_invocation":
                qualified_call = self._resolve_method_call(node, source, this_class, var_types)
                if qualified_call:
                    calls.add(qualified_call)
            
            for child in node.children:
                walk(child)
        
        walk(method_node)
        return sorted(calls)
    
    def _resolve_method_call(self, node, source: str, this_class: str, var_types: Dict[str, str]) -> Optional[str]:
        """Resolve a method call to its qualified name"""
        method_name_node = node.child_by_field_name("name")
        object_node = node.child_by_field_name("object")
        
        method_name = (
            source[method_name_node.start_byte:method_name_node.end_byte]
            if method_name_node else "unknown"
        )
        
        if not object_node:
            return f"{this_class}.{method_name}"
        
        obj_text = self._resolve_object_name(object_node, source).strip()
        
        if obj_text == "this":
            return f"{this_class}.{method_name}"
        elif obj_text in var_types:
            return f"{var_types[obj_text]}.{method_name}"
        elif "." in obj_text:
            root = obj_text.split(".", 1)[0]
            if root in var_types:
                return f"{var_types[root]}.{method_name}"
        elif obj_text and obj_text[0].isupper():
            return f"{obj_text}.{method_name}"
        
        return f"unknown.{method_name}"
    
    def _build_chunk(self, *, project_id: str, file_path: Path, class_name: str,
                    method_name: Optional[str], chunk_type: str, node,
                    source: str, calls: List[str], endpoints: List[Dict[str, str]],
                    extends: Optional[str] = None, implements: Optional[List[str]] = None,
                    vars: List[str] = None, is_job: bool = False,
                    job_info: Optional[JobInfo] = None) -> CodeChunk:
        """Build a code chunk with all metadata"""
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        chunk_id = f"{file_path}::{class_name}::{method_name or ''}::{line_start}::{line_end}"
        
        extends_list = [extends] if extends else []
        implements_list = implements or []
        vars_list = vars or []
        
        return {
            "id": chunk_id,
            "project_id": project_id,
            "file_path": str(file_path),
            "class_name": class_name,
            "method_name": method_name,
            "chunk_type": chunk_type,
            "calls": self._deduplicate(calls),
            "called_by": [],
            "line_start": line_start,
            "line_end": line_end,
            "content": source[node.start_byte:node.end_byte],
            "endpoints": self._deduplicate_endpoints(endpoints),
            "extends": self._deduplicate(extends_list),
            "implements": self._deduplicate(implements_list),
            "extended_by": [],
            "implemented_by": [],
            "vars": self._deduplicate(vars_list),
            "summary": "",
            "is_job": is_job,
            "job_info": asdict(job_info) if job_info else None,
            "publishes_to": [],
            "listens_to": [],
            "configures": [],
            "configured_by": []
        }
    
    def _deduplicate(self, items: List[str]) -> List[str]:
        """Remove duplicates while preserving order"""
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    def _deduplicate_endpoints(self, endpoints: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate endpoints"""
        seen = set()
        result = []
        for endpoint in endpoints:
            key = f"{endpoint.get('path', '')}:{endpoint.get('method', '')}"
            if key not in seen:
                seen.add(key)
                result.append(endpoint)
        return result
    
    # Additional utility methods for configuration extraction, relationship building, etc.
    # [Remaining implementation with consolidated patterns and optimizations]
    
    def _extract_schedule_config(self, annotation: str) -> Dict:
        """Extract scheduling configuration from @Scheduled annotation"""
        config = {}
        patterns = {
            'cron': r'cron\s*=\s*"([^"]+)"',
            'fixedRate': r'fixedRate\s*=\s*(\d+)',
            'fixedDelay': r'fixedDelay\s*=\s*(\d+)',
            'initialDelay': r'initialDelay\s*=\s*(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, annotation)
            if match:
                config[key] = int(match.group(1)) if key != 'cron' else match.group(1)
        
        return config
    
    def _extract_topics_from_annotation(self, annotation: str) -> List[str]:
        """Extract topics/queues from message listener annotations"""
        topics = []
        
        # Match topics = {"topic1", "topic2"} or topics = "topic1"
        topics_match = re.search(r'topics?\s*=\s*\{([^}]+)\}', annotation)
        if topics_match:
            topic_list = topics_match.group(1)
            topics.extend(re.findall(r'"([^"]+)"', topic_list))
        else:
            topic_match = re.search(r'topics?\s*=\s*"([^"]+)"', annotation)
            if topic_match:
                topics.append(topic_match.group(1))
        
        return topics
    
    def _extract_retry_config(self, annotation: str) -> Dict:
        """Extract retry configuration from @Retryable annotation"""
        config = {}
        patterns = {
            'maxAttempts': r'maxAttempts\s*=\s*(\d+)',
            'delay': r'delay\s*=\s*(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, annotation)
            if match:
                config[key] = int(match.group(1))
        
        return config

    def _remove_comments_from_source(self, source: str) -> str:
        """Remove all Java comments from source code"""
        # Remove single-line comments
        source = re.sub(r'//.*?$', '', source, flags=re.MULTILINE)
        # Remove multi-line and Javadoc comments
        source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
        
        # Clean up whitespace
        lines = [line for line in source.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_method_annotations(self, method_node, source: str) -> List[str]:
        """Extract all annotations from a method"""
        annotations = []
        for child in method_node.children:
            if child.type == "modifiers":
                for grandchild in child.children:
                    if grandchild.type == "annotation":
                        annotations.append(source[grandchild.start_byte:grandchild.end_byte])
        return annotations
    
    def _extract_vars(self, node) -> List[str]:
        """Extract variable type identifiers from a node"""
        query = "(type_identifier) @vars"
        try:
            captures = JAVA_LANGUAGE.query(query).captures(node)
            return [capture_node.text.decode() for capture_node, _ in captures]
        except:
            return []
    
    def _extract_method_signatures(self, class_node, source: str) -> List[Dict[str, str]]:
        """Extract method signatures from a class"""
        method_signatures = []
        method_nodes = self._get_method_nodes(class_node)
        
        for method_node in method_nodes:
            method_name = self._get_identifier(method_node, source) or "unknown"
            signature = self._extract_method_signature(method_node, source)
            method_signatures.append({"name": method_name, "signature": signature})
        
        return method_signatures
    
    def _extract_method_signature(self, method_node, source: str) -> str:
        """Extract method signature for comparison"""
        method_name = self._get_identifier(method_node, source) or "unknown"
        params = []
        
        for child in method_node.children:
            if child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "formal_parameter":
                        type_node = param.child_by_field_name("type")
                        if type_node:
                            param_type = source[type_node.start_byte:type_node.end_byte].strip()
                            param_type = param_type.split("<")[0]  # Remove generics
                            params.append(param_type)
        
        return f"{method_name}({','.join(params)})"
    
    def _resolve_method_inheritance(self, class_name: str, method_name: str, 
                                  method_signature: str) -> Dict[str, List[str]]:
        """Resolve which parent class/interface methods this method implements/overrides"""
        implements, extends = [], []
        
        if class_name not in self.class_hierarchy:
            return {"implements": implements, "extends": extends}
        
        current_class = self.class_hierarchy[class_name]
        
        # Check implemented interfaces
        for interface_name in current_class.implements:
            if interface_name in self.class_hierarchy:
                interface_methods = self.class_hierarchy[interface_name].methods
                for method_info in interface_methods:
                    if method_info["signature"] == method_signature:
                        implements.append(f"{interface_name}.{method_name}")
        
        # Check extended class
        if current_class.extends and current_class.extends in self.class_hierarchy:
            extended_methods = self.class_hierarchy[current_class.extends].methods
            for method_info in extended_methods:
                if method_info["signature"] == method_signature:
                    extends.append(f"{current_class.extends}.{method_name}")
        
        return {"implements": implements, "extends": extends}
    
    def _extract_class_hierarchy(self, class_node, source: str) -> Dict:
        """Extract class hierarchy (extends/implements) information"""
        extends = None
        implements = []
        
        # Extract superclass
        superclass_node = class_node.child_by_field_name("superclass")
        if superclass_node:
            type_id_node = (superclass_node.child_by_field_name("type_identifier") or
                          next((c for c in superclass_node.children if c.type == "type_identifier"), None))
            if type_id_node:
                extends = source[type_id_node.start_byte:type_id_node.end_byte]
        
        # Extract implemented interfaces
        interfaces_node = class_node.child_by_field_name("interfaces")
        if interfaces_node:
            type_list = next((c for c in interfaces_node.children if c.type == "type_list"), None)
            if type_list:
                for child in type_list.children:
                    if child.type == "type_identifier":
                        implements.append(source[child.start_byte:child.end_byte])
        
        return {"extends": extends, "implements": implements}
    
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
    
    def _extract_annotation_value(self, annotation_text: str, param_name: str) -> Optional[str]:
        """Extract value from annotation parameter"""
        pattern = rf'{param_name}\s*=\s*(?:"([^"]+)"|\{{?"([^"]+)"\}}?)'
        match = re.search(pattern, annotation_text)
        if match:
            return match.group(1) or match.group(2)
        
        # Handle short form (e.g., @GetMapping("/users"))
        if param_name == "value" and '"' in annotation_text:
            match = re.search(r'"([^"]+)"', annotation_text)
            return match.group(1) if match else None
        return None
    
    def _extract_method_endpoint(self, method_node, source: str) -> Optional[Tuple[str, str]]:
        """Extract endpoint information from method annotations"""
        method_path = ""
        http_method = "REQUEST"
        
        for child in method_node.children:
            if child.type == "modifiers":
                for grandchild in child.children:
                    if grandchild.type == "annotation":
                        text = source[grandchild.start_byte:grandchild.end_byte]
                        endpoint_info = self._parse_endpoint_annotation(text)
                        if endpoint_info:
                            method_path, http_method = endpoint_info
                            break
        
        if not method_path and http_method == "REQUEST":
            return None
        
        # Get class-level path
        class_path = self._get_class_level_path(method_node, source)
        full_path = self._combine_paths(class_path, method_path)
        
        return full_path or "/", http_method
    
    def _parse_endpoint_annotation(self, text: str) -> Optional[Tuple[str, str]]:
        """Parse endpoint annotation to extract path and method"""
        mappings = {
            "GetMapping": "GET",
            "PostMapping": "POST", 
            "PutMapping": "PUT",
            "DeleteMapping": "DELETE",
            "RequestMapping": "REQUEST"
        }
        
        for mapping, method in mappings.items():
            if f"@{mapping}" in text:
                if mapping == "RequestMapping":
                    method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', text)
                    if method_match:
                        method = method_match.group(1).upper()
                
                path = (self._extract_annotation_value(text, "value") or 
                       self._extract_annotation_value(text, "path") or "")
                return path, method
        
        return None
    
    def _get_class_level_path(self, method_node, source: str) -> str:
        """Get class-level RequestMapping path"""
        parent = method_node.parent
        while parent and parent.type != "class_declaration":
            parent = parent.parent
        
        if parent:
            for child in parent.children:
                if child.type == "modifiers":
                    for grandchild in child.children:
                        if grandchild.type == "annotation":
                            text = source[grandchild.start_byte:grandchild.end_byte]
                            if "@RequestMapping" in text:
                                return (self._extract_annotation_value(text, "value") or 
                                       self._extract_annotation_value(text, "path") or "")
        return ""
    
    def _combine_paths(self, class_path: str, method_path: str) -> str:
        """Combine class and method paths robustly"""
        if class_path and method_path:
            return (class_path.rstrip("/") + "/" + method_path.lstrip("/")).rstrip("/")
        return (method_path or class_path).rstrip("/")
    
    def _extract_fields(self, class_node, source: str) -> Dict[str, str]:
        """Extract field declarations from class"""
        field_map = {}
        
        for child in class_node.children:
            if child.type == "class_body":
                field_nodes = [n for n in child.children if n.type == "field_declaration"]
                for field_node in field_nodes:
                    self._process_field_declaration(field_node, source, field_map)
        
        return field_map
    
    def _process_field_declaration(self, field_node, source: str, field_map: Dict[str, str]):
        """Process a single field declaration"""
        type_node = field_node.child_by_field_name("type")
        if not type_node:
            return
        
        raw_type = source[type_node.start_byte:type_node.end_byte].strip()
        type_name = raw_type.split("<")[0].strip()  # Remove generics
        
        # Handle multiple declarators (e.g., String a, b;)
        declarator_nodes = [n for n in field_node.children if n.type == "variable_declarator"]
        for decl in declarator_nodes:
            name_node = decl.child_by_field_name("name")
            if name_node:
                var_name = source[name_node.start_byte:name_node.end_byte].strip()
                field_map[var_name] = type_name
    
    def _extract_param_types(self, method_node, source: str) -> Dict[str, str]:
        """Extract parameter types from method declaration"""
        param_map = {}
        for child in method_node.children:
            if child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "formal_parameter":
                        type_node = param.child_by_field_name("type")
                        name_node = param.child_by_field_name("name")
                        if type_node and name_node:
                            typename = source[type_node.start_byte:type_node.end_byte].strip()
                            name = source[name_node.start_byte:name_node.end_byte].strip()
                            param_map[name] = typename
        return param_map
    
    def _resolve_object_name(self, node, source: str) -> str:
        """Resolve the left-hand-side of a method call or field access"""
        if not node:
            return "unknown"
        
        if node.type == "identifier":
            return source[node.start_byte:node.end_byte]
        
        if node.type == "field_access":
            obj = self._resolve_object_name(node.child_by_field_name("object"), source)
            field = node.child_by_field_name("field")
            if field:
                return f"{obj}.{source[field.start_byte:field.end_byte]}"
            return obj
        
        if node.type == "method_invocation":
            return self._resolve_object_name(node.child_by_field_name("object"), source)
        
        return "unknown"
    
    # Relationship building methods
    
    def _populate_called_by(self, dep_graph: DependencyGraph):
        """Populate called_by relationships from calls"""
        for caller, relations in dep_graph.items():
            for callee in relations.get("calls", []):
                if callee in dep_graph:
                    dep_graph[callee].setdefault("called_by", [])
                    if caller not in dep_graph[callee]["called_by"]:
                        dep_graph[callee]["called_by"].append(caller)
    
    def _attach_called_by_to_chunks(self, chunks: List[CodeChunk], dep_graph: DependencyGraph):
        """Attach called_by information to chunks"""
        for chunk in chunks:
            if chunk["method_name"]:
                key = f"{chunk['class_name']}.{chunk['method_name']}"
                chunk["called_by"] = self._deduplicate(dep_graph.get(key, {}).get("called_by", []))
    
    def _populate_extends_and_implements_by(self, chunks: List[CodeChunk], dep_graph: DependencyGraph):
        """Populate reverse inheritance relationships"""
        # Build reverse relationships in graph
        for chunk in chunks:
            class_name = chunk.get("class_name", "")
            if not class_name:
                continue
            
            # Process implements relationships
            for parent in chunk.get("implements", []):
                dep_graph.setdefault(parent, {}).setdefault("implemented_by", [])
                if class_name not in dep_graph[parent]["implemented_by"]:
                    dep_graph[parent]["implemented_by"].append(class_name)
            
            # Process extends relationships
            for parent in chunk.get("extends", []):
                dep_graph.setdefault(parent, {}).setdefault("extended_by", [])
                if class_name not in dep_graph[parent]["extended_by"]:
                    dep_graph[parent]["extended_by"].append(class_name)
        
        # Update chunks with reverse relationships
        for chunk in chunks:
            class_name = chunk.get("class_name", "")
            if class_name in dep_graph:
                chunk["extended_by"] = self._deduplicate(dep_graph[class_name].get("extended_by", []))
                chunk["implemented_by"] = self._deduplicate(dep_graph[class_name].get("implemented_by", []))
    
    def _link_event_publishers_and_listeners(self, chunks: List[CodeChunk], dep_graph: DependencyGraph):
        """Link event publishers with their listeners"""
        for event_type, participants in self.event_registry.items():
            publishers, listeners = self._categorize_event_participants(chunks, participants)
            self._create_event_relationships(publishers, listeners, dep_graph, chunks, event_type)
    
    def _categorize_event_participants(self, chunks: List[CodeChunk], participants: List[str]) -> Tuple[List[str], List[str]]:
        """Categorize participants as publishers or listeners"""
        publishers, listeners = [], []
        
        for participant in participants:
            chunk = self._find_chunk_by_method(chunks, participant)
            if chunk:
                content = chunk.get("content", "").lower()
                if any(pattern.lower() in content for pattern in PATTERNS["event_publishers"]):
                    publishers.append(participant)
                elif any(pattern.lower() in content for pattern in PATTERNS["event_listeners"]):
                    listeners.append(participant)
        
        return publishers, listeners
    
    def _create_event_relationships(self, publishers: List[str], listeners: List[str], 
                                  dep_graph: DependencyGraph, chunks: List[CodeChunk], event_type: str):
        """Create relationships between publishers and listeners"""
        for publisher in publishers:
            for listener in listeners:
                if publisher != listener:
                    # Update dependency graph
                    dep_graph.setdefault(publisher, {}).setdefault("publishes_to", []).append(listener)
                    dep_graph.setdefault(listener, {}).setdefault("listens_to", []).append(publisher)
                    
                    # Update chunks
                    self._update_chunk_event_relationships(chunks, publisher, listener, "publishes_to")
                    self._update_chunk_event_relationships(chunks, listener, publisher, "listens_to")
                    
                    logger.info(f"Linked event: {publisher} -> {listener} (event: {event_type})")
    
    def _find_chunk_by_method(self, chunks: List[CodeChunk], method_name: str) -> Optional[CodeChunk]:
        """Find chunk by full method name"""
        return next((c for c in chunks if 
                    f"{c['class_name']}.{c['method_name']}" == method_name), None)
    
    def _update_chunk_event_relationships(self, chunks: List[CodeChunk], 
                                        source_method: str, target_method: str, relationship_type: str):
        """Update chunk with event relationship"""
        chunk = self._find_chunk_by_method(chunks, source_method)
        if chunk:
            chunk.setdefault(relationship_type, []).append(target_method)
    
    def _link_job_dependencies(self, chunks: List[CodeChunk], dep_graph: DependencyGraph):
        """Link jobs with their dependencies and configurations"""
        for job_name, job_info in self.job_registry.items():
            job_chunk = self._find_chunk_by_method(chunks, job_name)
            if job_chunk:
                job_chunk["job_info"] = asdict(job_info)
                job_chunk["is_job"] = True
                self._link_job_with_config(job_chunk, chunks, dep_graph, job_info)
    
    def _link_job_with_config(self, job_chunk: CodeChunk, chunks: List[CodeChunk], 
                            dep_graph: DependencyGraph, job_info: JobInfo):
        """Link job with its configuration classes"""
        class_name = job_chunk["class_name"]
        config_chunks = [c for c in chunks if c.get("chunk_type") == "configuration"]
        
        for config_chunk in config_chunks:
            if class_name in config_chunk.get("content", ""):
                self._create_job_config_relationship(job_chunk, config_chunk, dep_graph)
    
    def _create_job_config_relationship(self, job_chunk: CodeChunk, config_chunk: CodeChunk, 
                                      dep_graph: DependencyGraph):
        """Create bidirectional job-config relationship"""
        config_name = f"{config_chunk['class_name']}.{config_chunk.get('method_name', '')}"
        job_name = f"{job_chunk['class_name']}.{job_chunk['method_name']}"
        
        # Update dependency graph
        dep_graph.setdefault(config_name, {}).setdefault("configures", []).append(job_name)
        dep_graph.setdefault(job_name, {}).setdefault("configured_by", []).append(config_name)
        
        # Update chunks
        job_chunk.setdefault("configured_by", []).append(config_name)
        config_chunk.setdefault("configures", []).append(job_name)
        
        logger.info(f"Linked job config: {config_name} -> {job_name}")

# Export and utility functions

def generate_summary(chunks: List[CodeChunk], dep_graph: DependencyGraph) -> Dict:
    """Generate comprehensive summary statistics"""
    summary = {
        "total_chunks": len(chunks),
        "total_classes": len([c for c in chunks if c["method_name"] is None]),
        "total_methods": len([c for c in chunks if c["method_name"] is not None]),
        "total_jobs": len([c for c in chunks if c.get("is_job", False)]),
        "chunk_types": {},
        "job_types": {},
        "dependency_stats": {
            "total_nodes": len(dep_graph),
            "total_relationships": 0,
            "event_relationships": 0,
            "job_config_relationships": 0
        }
    }
    
    # Count chunk and job types
    for chunk in chunks:
        chunk_type = chunk["chunk_type"]
        summary["chunk_types"][chunk_type] = summary["chunk_types"].get(chunk_type, 0) + 1
        
        if chunk.get("is_job", False):
            job_info = chunk.get("job_info", {})
            job_type = job_info.get("type", "unknown")
            summary["job_types"][job_type] = summary["job_types"].get(job_type, 0) + 1
    
    # Count relationships
    for node_data in dep_graph.values():
        for rel_type in ["calls", "called_by", "extends", "implements", "extended_by", "implemented_by"]:
            summary["dependency_stats"]["total_relationships"] += len(node_data.get(rel_type, []))
        
        # Count special relationships
        summary["dependency_stats"]["event_relationships"] += len(node_data.get("publishes_to", []))
        summary["dependency_stats"]["job_config_relationships"] += len(node_data.get("configures", []))
    
    return summary

def export_chunks_to_json(chunks: List[CodeChunk], output_json_path: str):
    """Export chunks to JSON file"""
    serializable_chunks = []
    for chunk in chunks:
        serializable_chunk = dict(chunk)
        # Ensure JSON serialization
        for key, value in serializable_chunk.items():
            if isinstance(value, set):
                serializable_chunk[key] = list(value)
        serializable_chunks.append(serializable_chunk)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Exported {len(chunks)} chunks to {output_json_path}")

def export_job_report(chunks: List[CodeChunk], output_path: str) -> Dict:
    """Export detailed job report"""
    jobs = [chunk for chunk in chunks if chunk.get("is_job", False)]
    
    report = {
        "summary": {"total_jobs": len(jobs), "job_types": {}},
        "jobs": []
    }
    
    for job in jobs:
        job_info = job.get("job_info", {})
        job_type = job_info.get("type", "unknown")
        
        report["summary"]["job_types"][job_type] = report["summary"]["job_types"].get(job_type, 0) + 1
        
        job_detail = {
            "class_name": job["class_name"],
            "method_name": job["method_name"],
            "file_path": job["file_path"],
            "job_type": job_type,
            "schedule": job_info.get("schedule"),
            "async": job_info.get("async", False),
            "event_types": job_info.get("event_types", []),
            "topics": job_info.get("topics", []),
            "retry_config": job_info.get("retry_config", {}),
            "publishes_to": job.get("publishes_to", []),
            "listens_to": job.get("listens_to", []),
            "configured_by": job.get("configured_by", []),
            "calls": job.get("calls", []),
            "called_by": job.get("called_by", [])
        }
        report["jobs"].append(job_detail)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Exported job report to {output_path}")
    return report

def export_chunks_to_pyvis_html(chunks: List[CodeChunk], dep_graph: DependencyGraph, output_html_path: str):
    """Export dependency graph to interactive HTML visualization"""
    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.barnes_hut()
    
    # Color mapping for different types
    color_map = {
        "job": "#ff6b6b", "controller": "#3b82f6", "service": "#10b981",
        "repository": "#f59e0b", "configuration": "#8b5cf6", "default": "#6b7280",
        "scheduled": "#dc2626", "event_listener": "#ea580c", 
        "async": "#c2410c", "method": "#22c55e"
    }
    
    added_nodes = set()
    
    # Add nodes
    for chunk in chunks:
        node_id, label, color, title = _prepare_node_data(chunk, color_map)
        net.add_node(node_id, label=label, color=color, title=title)
        added_nodes.add(node_id)
    
    # Add edges with different colors for different relationships
    edge_colors = {
        "calls": "#374151", "extends": "#f59e0b", "implements": "#a21caf",
        "publishes_event": "#ef4444", "configures": "#8b5cf6"
    }
    
    for node, relations in dep_graph.items():
        if node not in added_nodes:
            continue
        
        # Add various relationship edges
        _add_relationship_edges(net, node, relations, added_nodes, edge_colors)
    
    # Configure visualization
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      },
      "interaction": {
        "multiselect": true,
        "selectConnectedEdges": false
      }
    }
    """)
    
    net.write_html(output_html_path)
    logger.info(f"Exported dependency graph to {output_html_path}")

def _prepare_node_data(chunk: CodeChunk, color_map: Dict[str, str]) -> Tuple[str, str, str, str]:
    """Prepare node data for visualization"""
    if chunk["method_name"] is None:
        node_id = chunk["class_name"]
        label = f"Class: {chunk['class_name']}"
        color = color_map.get(chunk["chunk_type"], color_map["default"])
        if chunk.get("is_job", False):
            color = color_map["job"]
    else:
        node_id = f"{chunk['class_name']}.{chunk['method_name']}"
        label = f"{chunk['class_name']}.{chunk['method_name']}()"
        if chunk.get("is_job", False):
            job_type = chunk.get("job_info", {}).get("type", "job")
            color = color_map.get(job_type, color_map["job"])
        else:
            color = color_map["method"]
    
    # Create title with job info if applicable
    title = chunk.get("content", "")[:500]
    if chunk.get("is_job", False):
        job_info = chunk.get("job_info", {})
        title = f"JOB: {job_info.get('type', 'unknown')}\n{title}..."
    
    return node_id, label, color, title

def _add_relationship_edges(net: Network, node: str, relations: Dict, 
                          added_nodes: Set[str], edge_colors: Dict[str, str]):
    """Add relationship edges to the network"""
    edge_mappings = [
        ("calls", "calls", 1),
        ("extends", "extends", 2),
        ("implements", "implements", 2),
        ("publishes_to", "publishes_event", 3),
        ("configures", "configures", 2)
    ]
    
    for rel_type, edge_label, width in edge_mappings:
        for target in relations.get(rel_type) or []:
            if target in added_nodes:
                color = edge_colors.get(edge_label, "#374151")
                net.add_edge(node, target, color=color, title=edge_label, width=width)

# Main entry point for backward compatibility
def parse_project(root: Path, project_id: str, remove_comments: bool = True) -> Tuple[List[CodeChunk], DependencyGraph]:
    """Main entry point function for backward compatibility"""
    analyzer = JavaCodeAnalyzer(remove_comments)
    return analyzer.parse_project(root, project_id)