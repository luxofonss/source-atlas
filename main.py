from parser import parse_project,export_chunks_to_pyvis_html,export_chunks_to_json
from pathlib import Path

chunks, dep_graph = parse_project(Path("F:/_side_projects/source_atlas/data/repo/onestudy-server"), "main", True)
export_chunks_to_pyvis_html(chunks, dep_graph, "output_graph.html")
export_chunks_to_json(chunks, "output_chunks.json")

print("hello world")