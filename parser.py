from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from loguru import logger
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from utils.constant import JAVA_STANDARD_TYPES, GENERIC_TYPE_VARS
from pyvis.network import Network

JAVA_LANGUAGE: Language = get_language("java")
_PARSER = Parser()
_PARSER.set_language(JAVA_LANGUAGE)
EXCLUDE_CHUNK_TYPE = ["package_declaration", "import_declaration"]

CodeChunk = Dict[str, object]
DependencyGraph = Dict[str, Dict[str, List[str]]]

BLACKLIST_DIR = {
    ".git", ".idea", "env", ".github", ".gitlab", "target", "build", 
    "out", "bin", ".vscode", "node_modules", "__pycache__"
}
WHITELIST_EXT = {".java"}


def _should_skip(path: Path) -> bool:
    parts = {p.name for p in path.parents}
    return bool(parts & BLACKLIST_DIR)

def _list_java_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.java"):
        if p.suffix in WHITELIST_EXT and not _should_skip(p):
            files.append(p)
    return files

def _check_duplicate_chunks(chunks: List[CodeChunk]) -> List[CodeChunk]:
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
            logger.info(f"Duplicate chunk found with ID: {chunk_id}")
    
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate chunks")
    
    return unique_chunks

def parse_project(root: Path, project_id: str, remove_comments: bool = True) -> Tuple[List[CodeChunk], DependencyGraph]:
    chunks: List[CodeChunk] = []
    dep_graph: DependencyGraph = {}
    class_hierarchy: Dict[str, Dict[str, any]] = {}

    java_files = _list_java_files(root)
    
    # First pass: collect class hierarchy information
    for file_path in java_files:
        try:
            text = file_path.read_text("utf-8")
            if remove_comments:
                text = _remove_comments(text)
            tree = _PARSER.parse(text.encode("utf-8"))
            _collect_class_hierarchy(file_path, tree, text, class_hierarchy)
        except Exception as exc:
            logger.error(f"Error collecting hierarchy for {file_path}: {exc}")
    
    # Second pass: parse files with hierarchy context
    for i, file_path in enumerate(java_files):
        logger.info(f"Processing file {i+1}/{len(java_files)}: {file_path}")
        
        try:
            text = file_path.read_text("utf-8")
        except UnicodeDecodeError:
            logger.info(f"Could not read {file_path} – skipping")
            continue

        # Remove comments before parsing if requested
        if remove_comments:
            logger.info(f"Removing comments from {file_path}")
            cleaned_text = _remove_comments(text)
        else:
            cleaned_text = text

        try:
            tree = _PARSER.parse(cleaned_text.encode("utf-8"))
        except Exception as exc:
            logger.error(f"Parser error for {file_path}: {exc}")
            continue

        file_chunks, file_graph = _parse_file(project_id, file_path, tree, cleaned_text, class_hierarchy)
        chunks.extend(file_chunks)
        dep_graph.update(file_graph)

    # Check for duplicate chunks
    chunks = _check_duplicate_chunks(chunks)
    
    _populate_called_by(dep_graph)
    _attach_called_by_to_chunks(chunks, dep_graph)
    _populate_extends_and_implements_by(chunks, dep_graph)

    logger.info(f"Parse complete: {len(chunks)} chunks, {len(dep_graph)} graph nodes")
    return chunks, dep_graph


def _collect_class_hierarchy(file_path: Path, tree, source: str, class_hierarchy: Dict[str, Dict]):
    def _collect_all_classes(node):
        result = []
        if node.type == "class_declaration":
            result.append(node)
        for child in node.children:
            result.extend(_collect_all_classes(child))
        return result

    for class_node in _collect_all_classes(tree.root_node):
        class_name = _get_identifier(class_node, source) or file_path.stem
        hierarchy = _extract_class_hierarchy(class_node, source)
        
        class_hierarchy[class_name] = {
            "extends": hierarchy["extends"],
            "implements": hierarchy["implements"],
            "methods": _extract_method_signatures(class_node, source),
            "file_path": str(file_path)
        }

def _extract_method_signatures(class_node, source: str) -> List[Dict[str, str]]:
    """Extract method signatures from a class"""
    method_signatures = []
    
    method_nodes = []
    for child in class_node.children:
        if child.type in ["class_body", "interface_body", "enum_body"]:
            method_nodes.extend([
                n for n in child.children if n.type == "method_declaration"
            ])
    
    for method_node in method_nodes:
        method_name = _get_identifier(method_node, source) or "unknown"
        
        # Extract method signature (return type + parameters)
        signature = _extract_method_signature(method_node, source)
        method_signatures.append({
            "name": method_name,
            "signature": signature
        })
    
    return method_signatures


def _extract_method_signature(method_node, source: str) -> str:
    """Extract method signature for comparison"""
    # Get method name
    method_name = _get_identifier(method_node, source) or "unknown"
    
    # Get parameters
    params = []
    for child in method_node.children:
        if child.type == "formal_parameters":
            for param in child.children:
                if param.type == "formal_parameter":
                    type_node = param.child_by_field_name("type")
                    if type_node:
                        param_type = source[type_node.start_byte:type_node.end_byte].strip()
                        # Remove generic parameters for matching
                        param_type = param_type.split("<")[0]
                        params.append(param_type)
    
    return f"{method_name}({','.join(params)})"


def _resolve_method_inheritance(class_name: str, method_name: str, method_signature: str, 
                               class_hierarchy: Dict[str, Dict]) -> Dict[str, List[str]]:
    implements = []
    extends = []
    
    if class_name not in class_hierarchy:
        return {"implements": implements, "extends": extends}
    
    current_class = class_hierarchy[class_name]
    
    # Check implemented interfaces
    for interface_name in current_class.get("implements", []):
        if interface_name in class_hierarchy:
            interface_methods = class_hierarchy[interface_name].get("methods", [])
            for method_info in interface_methods:
                if method_info["signature"] == method_signature:
                    implements.append(f"{interface_name}.{method_name}")
                    logger.info(f"Method {class_name}.{method_name} implements {interface_name}.{method_name}")
    
    # Check extended class
    extended_class = current_class.get("extends")
    if extended_class and extended_class in class_hierarchy:
        extended_methods = class_hierarchy[extended_class].get("methods", [])
        for method_info in extended_methods:
            if method_info["signature"] == method_signature:
                extends.append(f"{extended_class}.{method_name}")
                logger.info(f"Method {class_name}.{method_name} overrides {extended_class}.{method_name}")
    
    return {"implements": implements, "extends": extends}


def _parse_file(project_id: str, file_path: Path, tree, source: str, class_hierarchy: Dict[str, Dict]) -> Tuple[List[CodeChunk], DependencyGraph]:
    """Parse a single Java file with class hierarchy context"""
    chunks: List[CodeChunk] = []
    graph: DependencyGraph = {}

    root_node = tree.root_node

    for class_node in [n for n in root_node.children if n.type not in EXCLUDE_CHUNK_TYPE]:
        class_name = _get_identifier(class_node, source) or file_path.stem
        chunk_type = _infer_chunk_type(class_node, source)
        class_endpoints = _extract_class_level_endpoints(class_node, source)
        class_hierarchy_info = _extract_class_hierarchy(class_node, source)

        field_map = _extract_fields(class_node, source)

        method_nodes = []
        for child in class_node.children:
            if child.type in ["class_body", "interface_body", "enum_body"]:
                method_nodes.extend([
                    n for n in child.children if n.type == "method_declaration"
                ])
    
        logger.info(f"Processing class: {class_name} with {len(method_nodes)} methods")
        
        # Create class-level chunk
        chunk = _build_chunk(
            project_id=project_id,
            file_path=file_path,
            class_name=class_name,
            method_name=None,
            chunk_type=chunk_type,
            start_point=class_node.start_point,
            end_point=class_node.end_point,
            content=source[class_node.start_byte:class_node.end_byte],
            calls=[],
            endpoints=[{"path": ep, "method": "REQUEST"} for ep in class_endpoints] if chunk_type == "controller" else [],
            extends=class_hierarchy_info['extends'],
            implements=class_hierarchy_info['implements'],
        )
        chunks.append(chunk)
        graph[f"{class_name}"] = {
            "calls": [], 
            "called_by": [], 
            "extends": class_hierarchy_info['extends'], 
            "implements": class_hierarchy_info['implements'],
            "extended_by": [],
            "implemented_by": []
        }

        # Process methods
        for m_node in method_nodes:
            method_name = _get_identifier(m_node, source) or "unknown"
            method_signature = _extract_method_signature(m_node, source)
            param_map = _extract_param_types(m_node, source)
            vars_list = _extract_vars(m_node)
            vars_list = [var for var in vars_list if var not in JAVA_STANDARD_TYPES and var not in GENERIC_TYPE_VARS]
            calls = _extract_calls(m_node, source, class_name, {**field_map, **param_map})

            # Resolve method-level inheritance
            method_inheritance = _resolve_method_inheritance(class_name, method_name, method_signature, class_hierarchy)

            endpoint = _extract_method_endpoint(m_node, source)
            endpoints = []
            if endpoint:
                path, method = endpoint
                endpoints.append({"path": path, "method": method})

            chunk = _build_chunk(
                project_id=project_id,
                file_path=file_path,
                class_name=class_name,
                method_name=method_name,
                chunk_type=chunk_type,
                start_point=m_node.start_point,
                end_point=m_node.end_point,
                content=source[m_node.start_byte:m_node.end_byte],
                calls=calls,
                endpoints=endpoints,
                extends=method_inheritance["extends"],
                implements=method_inheritance["implements"],
                vars=vars_list,
            )
            chunks.append(chunk)
            graph[f"{class_name}.{method_name}"] = {"calls": calls, "called_by": []}

    return chunks, graph


def _extract_vars(node) -> List[str]:
    """Extract variable type identifiers from a node"""
    q = """
    (
        (type_identifier) @vars
    )
    """
    try:
        captures = JAVA_LANGUAGE.query(q).captures(node)
        res = []
        for capture_node, capture_name in captures:
            res.append(capture_node.text.decode())
        return res
    except:
        return []


def _get_identifier(node, source: str) -> str | None:
    """Get identifier from a node"""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte]
    cursor = node.walk()
    while True:
        if cursor.node.type == "identifier":
            return source[cursor.node.start_byte:cursor.node.end_byte]
        if not cursor.goto_next_sibling():
            break
    return None


def _extract_calls(method_node, source: str, this_class: str, var_types: Dict[str, str]) -> List[str]:
    """Extract method calls from a method node"""
    calls: set[str] = set()

    def walk(node):
        if node.type == "method_invocation":
            method_name_node = node.child_by_field_name("name")
            object_node = node.child_by_field_name("object")

            method_name = (
                source[method_name_node.start_byte:method_name_node.end_byte]
                if method_name_node else "unknown"
            )

            qualified = f"unknown.{method_name}"  # fallback

            if object_node:
                obj_text = _resolve_object_name(object_node, source).strip()

                if obj_text == "this":
                    qualified = f"{this_class}.{method_name}"
                elif obj_text in var_types:
                    qualified = f"{var_types[obj_text]}.{method_name}"
                elif "." in obj_text:
                    # Try to resolve root var in a chain like repo.a.b -> repo
                    root = obj_text.split(".", 1)[0]
                    if root in var_types:
                        qualified = f"{var_types[root]}.{method_name}"
                    else:
                        logger.info(f"Unresolved object: '{obj_text}' -> '{method_name}'")
                        qualified = f"unknown.{method_name}"
                elif obj_text[0].isupper():
                    qualified = f"{obj_text}.{method_name}"
                else:
                    # logger.info(f"Unknown object: '{obj_text}' in method '{method_name}'")
                    qualified = f"unknown.{method_name}"
            else:
                qualified = f"{this_class}.{method_name}"

            calls.add(qualified)

        for child in node.children:
            walk(child)

    walk(method_node)
    return sorted(calls)


def _resolve_object_name(node, source: str) -> str:
    """Resolve the left-hand-side of a method call or field access"""
    if node is None:
        return "unknown"

    if node.type == "identifier":
        return source[node.start_byte:node.end_byte]

    if node.type == "field_access":
        obj = _resolve_object_name(node.child_by_field_name("object"), source)
        field = node.child_by_field_name("field")
        if field:
            return f"{obj}.{source[field.start_byte:field.end_byte]}"
        return obj

    if node.type == "method_invocation":
        return _resolve_object_name(node.child_by_field_name("object"), source)

    return "unknown"


def _extract_param_types(method_node, source: str) -> Dict[str, str]:
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


def _infer_chunk_type(node, source: str) -> str:
     """Infer the type of a code chunk based on annotations"""
     annotations = [c for c in node.children if c.type == "modifiers"]
     ann_text = "".join(source[c.start_byte:c.end_byte].lower() for c in annotations)

     if "@controller" in ann_text or "@restcontroller" in ann_text:
          return "controller"
     if "@service" in ann_text:
          return "service"
     if "@repository" in ann_text:
          return "repository"
     if "@entity" in ann_text:
          return "entity"
     if "filter" in ann_text:
          return "filter"
     if "component" in ann_text:
          return "component"
     if "bean" in ann_text:
          return "bean"
     if "abstract" in ann_text:
          return "abstract_class"
     if "scheduled" in ann_text:
          return "scheduled"
     if "configuration" in ann_text or "controlleradvice" in ann_text:
          return "configuration"

     if node.type == "interface_declaration":
          return "interface"

     return "other"


def _extract_class_level_endpoints(class_node, source: str) -> List[str]:
    """Extract endpoints from class-level RequestMapping annotations"""
    paths = []
    for child in class_node.children:
        if child.type == "modifiers":
            text = source[child.start_byte:child.end_byte]
            if "@RequestMapping" in text:
                val = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path")
                if val:
                    paths.append(val)
    return paths


def _extract_annotation_value(annotation_text: str, param_name: str) -> str | None:
    """Extract value from annotation parameter"""
    # Handle value = "/path", path = "/path", or short form like @GetMapping("/users")
    pattern = rf'{param_name}\s*=\s*(?:"([^"]+)"|\{{?"([^"]+)"\}}?)'
    match = re.search(pattern, annotation_text)
    if match:
        return match.group(1) or match.group(2)
    # Handle short form (e.g., @GetMapping("/users"))
    if param_name == "value" and '"' in annotation_text:
        match = re.search(r'"([^"]+)"', annotation_text)
        return match.group(1) if match else None
    return None


def _extract_method_endpoint(method_node, source: str) -> Tuple[str, str] | None:
    """Extract endpoint information from method annotations"""
    method_path = ""
    http_method = "REQUEST"

    for child in method_node.children:
        if child.type == "modifiers":
            for grandchild in child.children:
                if grandchild.type == "annotation":
                    text = source[grandchild.start_byte:grandchild.end_byte]
                    for ann in ["GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "RequestMapping"]:
                        if f"@{ann}" in text:
                            if ann != "RequestMapping":
                                http_method = ann.replace("Mapping", "").upper()
                            else:
                                # Check for method parameter in RequestMapping
                                method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', text)
                                if method_match:
                                    http_method = method_match.group(1).upper()
                            method_path = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path") or ""
                            break

    if not method_path and http_method == "REQUEST":
        return None

    class_path = ""
    parent = method_node.parent
    while parent:
        if parent.type == "class_declaration":
            for child in parent.children:
                if child.type == "modifiers":
                    for grandchild in child.children:
                        if grandchild.type == "annotation":
                            text = source[grandchild.start_byte:grandchild.end_byte]
                            if "@RequestMapping" in text:
                                class_path = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path") or ""
            break
        parent = parent.parent

    # Robust path combination
    if class_path and method_path:
        full_path = (class_path.rstrip("/") + "/" + method_path.lstrip("/")).rstrip("/")
    elif method_path:
        full_path = method_path.rstrip("/")
    else:
        full_path = class_path.rstrip("/")
    return full_path or "/", http_method


def _extract_fields(class_node, source: str) -> Dict[str, str]:
    """Extract field declarations from class"""
    field_map: Dict[str, str] = {}

    field_nodes = []
    for child in class_node.children:
        if child.type == "class_body":
            field_nodes.extend([
                n for n in child.children if n.type == "field_declaration"
            ])

    for field_node in field_nodes:
        type_node = field_node.child_by_field_name("type")
        if not type_node:
            continue

        # Extract base type (ignore generic parameters)
        raw_type = source[type_node.start_byte:type_node.end_byte].strip()
        type_name = raw_type.split("<")[0].strip()

        # Support multiple declarators (e.g., `String a, b;`)
        declarator_nodes = [
            n for n in field_node.children if n.type == "variable_declarator"
        ]
        for decl in declarator_nodes:
            name_node = decl.child_by_field_name("name")
            if name_node:
                var_name = source[name_node.start_byte:name_node.end_byte].strip()
                field_map[var_name] = type_name

    return field_map


def _deduplicate_list(data: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order"""
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def _deduplicate_endpoints(endpoints: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate endpoints"""
    seen = set()
    result = []
    for endpoint in endpoints:
        # Create a unique key for each endpoint
        key = f"{endpoint.get('path', '')}:{endpoint.get('method', '')}"
        if key not in seen:
            seen.add(key)
            result.append(endpoint)
    return result

def _build_chunk(
    *,
    project_id: str,
    file_path: Path,
    class_name: str,
    method_name: str | None,
    chunk_type: str,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    content: str,
    calls: List[str],
    endpoints: List[Dict[str, str]],
    extends: Optional[str | List[str]] = None,
    implements: Optional[List[str]] = None,
    vars: List[str] = []
) -> CodeChunk:
    """Build a code chunk with all metadata"""
    line_start = start_point[0] + 1
    line_end = end_point[0] + 1
    chunk_id = f"{file_path}::{class_name}::{method_name or ''}::{line_start}::{line_end}"
    
    # Handle extends being either string or list
    if isinstance(extends, str):
        extends_list = [extends] if extends else []
    elif isinstance(extends, list):
        extends_list = extends
    else:
        extends_list = []
    
    # Deduplicate all list fields
    calls = _deduplicate_list(calls)
    vars = _deduplicate_list(vars)
    implements = _deduplicate_list(implements or [])
    extends_list = _deduplicate_list(extends_list)
    endpoints = _deduplicate_endpoints(endpoints)
    
    return {
        "id": chunk_id,
        "project_id": project_id,
        "file_path": str(file_path),
        "class_name": class_name,
        "method_name": method_name,
        "chunk_type": chunk_type,
        "calls": calls,
        "called_by": [],
        "line_start": line_start,
        "line_end": line_end,
        "content": content,
        "endpoints": endpoints,
        "extends": extends_list,
        "implements": implements,
        "extended_by": [],
        "implemented_by": [],
        "vars": vars,
        "summary": ""  # Can be populated later with AI summarization
    }


def _populate_called_by(dep_graph: DependencyGraph):
    """Populate called_by relationships from calls"""
    for caller, rel in dep_graph.items():
        for callee in rel.get("calls", []):
            if callee in dep_graph:
                # Deduplicate called_by list
                if caller not in dep_graph[callee].get("called_by", []):
                    dep_graph[callee].setdefault("called_by", []).append(caller)


def _attach_called_by_to_chunks(chunks: List[CodeChunk], dep_graph: DependencyGraph):
    """Attach called_by information to chunks"""
    for chunk in chunks:
        if chunk["method_name"] is None:
            continue
        key = f"{chunk['class_name']}.{chunk['method_name']}"
        chunk["called_by"] = _deduplicate_list(dep_graph.get(key, {}).get("called_by", []))


def _populate_extends_and_implements_by(chunks: List[CodeChunk], dep_graph: DependencyGraph):
    """Populate extends and implements relationships"""
    # First, populate the graph relationships
    for chunk in chunks:
        this_class = str(chunk.get("class_name", ""))
        if not this_class:
            continue

        implements = chunk.get("implements", [])
        if isinstance(implements, list):
            for parent in implements:
                dep_graph.setdefault(str(parent), {}).setdefault("implemented_by", [])
                if this_class not in dep_graph[str(parent)]["implemented_by"]:
                    dep_graph[str(parent)]["implemented_by"].append(this_class)

        extends = chunk.get("extends", [])
        if isinstance(extends, list):
            for parent in extends:
                dep_graph.setdefault(str(parent), {}).setdefault("extended_by", [])
                if this_class not in dep_graph[str(parent)]["extended_by"]:
                    dep_graph[str(parent)]["extended_by"].append(this_class)
    
    # Then, populate the chunk relationships
    for chunk in chunks:
        class_name = str(chunk.get("class_name", ""))
        if class_name in dep_graph:
            chunk["extended_by"] = _deduplicate_list(dep_graph[class_name].get("extended_by", []))
            chunk["implemented_by"] = _deduplicate_list(dep_graph[class_name].get("implemented_by", []))

def _extract_class_hierarchy(class_node, source: str) -> dict:
    """Extract class hierarchy (extends/implements) information"""
    extends = None
    implements = []

    # Extract superclass
    superclass_node = class_node.child_by_field_name("superclass")
    if superclass_node:
        type_id_node = superclass_node.child_by_field_name("type_identifier")
        if not type_id_node:
            type_id_node = next(
                (child for child in superclass_node.children if child.type == "type_identifier"),
                None
            )
        if type_id_node:
            extends = source[type_id_node.start_byte:type_id_node.end_byte]

    # Extract mplemented interfaces
    interfaces_node = class_node.child_by_field_name("interfaces")
    if interfaces_node:
        type_list = next(
            (child for child in interfaces_node.children if child.type == "type_list"),
            None
        )
        if type_list:
            for child in type_list.children:
                if child.type == "type_identifier":
                    implements.append(source[child.start_byte:child.end_byte])

    return {
        "extends": extends,
        "implements": implements
    }

def generate_summary(chunks: List[CodeChunk], dep_graph: DependencyGraph) -> Dict:
    """Generate summary statistics"""
    summary = {
        "total_chunks": len(chunks),
        "total_classes": len([c for c in chunks if c["method_name"] is None]),
        "total_methods": len([c for c in chunks if c["method_name"] is not None]),
        "chunk_types": {},
        "dependency_stats": {
            "total_nodes": len(dep_graph),
            "total_relationships": 0
        }

    }

    # Count chunk types
    for chunk in chunks:
        chunk_type = chunk["chunk_type"]
        summary["chunk_types"][chunk_type] = summary["chunk_types"].get(chunk_type, 0) + 1

    # Count relationships
    for node_data in dep_graph.values():
        for rel_type in ["calls", "called_by", "extends", "implements", "extended_by", "implemented_by"]:
            if rel_type in node_data:
                summary["dependency_stats"]["total_relationships"] += len(node_data[rel_type])


    return summary

def _remove_comments(source: str) -> str:
    """Remove all Java comments from source code"""
    import re

    source = re.sub(r'//.*?$', '', source, flags=re.MULTILINE)
    # Remove multi-line comments (/* ... */)
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    # Remove Javadoc comments (/** ... */)
    source = re.sub(r'/\*\*.*?\*/', '', source, flags=re.DOTALL)
    # Clean up extra whitespace and empty lines
    lines = source.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:  # Keep non-empty lines
            cleaned_lines.append(line)
        else:
            # Keep empty lines but reduce multiple consecutive empty lines
            if not cleaned_lines or cleaned_lines[-1].strip():
                cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def export_chunks_to_pyvis_html(chunks: List[CodeChunk], dep_graph: DependencyGraph, output_html_path: str):

    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # Add nodes
    added_nodes = set()
    for chunk in chunks:
        if chunk["method_name"] is None:
            node_id = chunk["class_name"]
            label = f"Class: {chunk['class_name']}"
            color = "#3b82f6"  # blue
        else:
            node_id = f"{chunk['class_name']}.{chunk['method_name']}"
            label = f"{chunk['class_name']}.{chunk['method_name']}()"
            color = "#22c55e"  # green
        net.add_node(node_id, label=label, color=color, title=chunk.get("content", ""))
        added_nodes.add(node_id)

    # Add edges
    for node, rel in dep_graph.items():
        # Calls
        for callee in rel.get("calls") or []:
            if callee in added_nodes:
                net.add_edge(node, callee, color="#222", title="calls")
        # Extends
        for parent in rel.get("extends") or []:
            if parent in added_nodes:
                net.add_edge(node, parent, color="#f59e42", title="extends")
        # Implements
        for parent in rel.get("implements") or []:
            if parent in added_nodes:
                net.add_edge(node, parent, color="#a21caf", title="implements")

    # Xuất file HTML
    net.write_html(output_html_path)

def export_chunks_to_json(chunks: List[CodeChunk], output_json_path: str):
    """
    Export the list of chunks to a JSON file.
    """
    import json
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)