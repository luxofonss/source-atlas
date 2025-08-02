from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Set, Optional

from tree_sitter import Language, Parser, Node
from models.domain_models import CodeChunk
from models.analyzer_config import AnalyzerConfig

class BaseFileProcessor(ABC):
    """Abstract base class for processing source code files of different languages."""
    
    def __init__(self, config: AnalyzerConfig, language: Language, parser: Parser):
        self.config = config
        self.language = language
        self.parser = parser
    
    @abstractmethod
    def process_file(self, file_path: Path, project_id: str, class_cache: Set[str] = None) -> List[CodeChunk]:
        """Process a single source file and return code chunks."""
        pass
    
    @abstractmethod
    def _read_file_content(self, file_path: Path) -> str:
        """Read the content of a file with appropriate encoding."""
        pass
    
    @abstractmethod
    def _extract_package(self, root_node: Node, content: str) -> str:
        """Extract package or module declaration."""
        pass
    
    @abstractmethod
    def _extract_imports(self, root_node: Node, content: str) -> dict:
        """Extract import statements."""
        pass
    
    @abstractmethod
    def _parse_class_node(self, class_node: Node, content: str, package: str, 
                         imports: dict, file_path: str, root_node: Node) -> Optional[CodeChunk]:
        """Parse a single class/module node."""
        pass