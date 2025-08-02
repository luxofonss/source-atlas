import logging
from pathlib import Path

from factory.analyzer_factory import AnalyzerFactory
from factory.config_builder import AnalyzerConfigBuilder
from models.domain_models import CodeChunk, DependencyGraph

logger = logging.getLogger(__name__)

def main():
    """Example usage of the multi-language code analyzer."""
    args = {
        "project_path": "F:/_side_projects/source_atlas/data/repo/test",
        "project_id": "test",
        "output": "./result",
        "language": "java",  # Can be changed to 'python' or others
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
        logger.info(f"Analyzing {args['language'].capitalize()} project at: {project_path}")
        
        config = AnalyzerConfigBuilder().with_comment_removal(args["remove_comments"]).build()
        analyzer = AnalyzerFactory.create_analyzer(args["language"], config)
        chunks, dependency_graph = analyzer.parse_project(project_path, args["project_id"])
        
        # Display results (same as original)
        logger.info(f"\nAnalysis Results:")
        logger.info(f"Found {len(chunks)} classes/interfaces/enums")
        logger.info(f"Dependency graph has {len(dependency_graph.nodes)} nodes and {len(dependency_graph.edges)} edges")
        
        if args["output"]:
            output_path = Path(args["output"])
            analyzer.export_results(chunks, dependency_graph, output_path)
        
        stats = analyzer.generate_statistics(chunks, dependency_graph)
        logger.info(f"\nProject Statistics:")
        logger.info(f"Total Classes: {stats['total_classes']}")
        logger.info(f"Total Methods: {stats['total_methods']}")
        logger.info(f"Total Fields: {stats['total_fields']}")
        
        logger.info(f"\nAnalysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())