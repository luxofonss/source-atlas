# from parser import parse_project,export_chunks_to_pyvis_html,export_chunks_to_json
from pathlib import Path
from parser_2 import parse_project

chunks, dep_graph = parse_project(Path("F:/_side_projects/source_atlas/data/repo/test"), "main", True)
# export_chunks_to_pyvis_html(chunks, dep_graph, "output_graph.html")
# export_chunks_to_json(chunks, "output_chunks.json")

print("hello world")