from pathlib import Path
from typing import Optional

from analyzers.java_analyzer import JavaCodeAnalyzer
from analyzers.base_analyzer import BaseCodeAnalyzer
from models.analyzer_config import AnalyzerConfig
from factory.config_builder import AnalyzerConfigBuilder

class AnalyzerFactory:
    """Factory for creating language-specific code analyzers."""
    
    @staticmethod
    def create_analyzer(language: str, config: Optional[AnalyzerConfig] = None) -> BaseCodeAnalyzer:
        """Create an analyzer for the specified language."""
        language = language.lower()
        if language == 'java':
            return JavaCodeAnalyzer(config or AnalyzerConfig())
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    @staticmethod
    def create_default_analyzer() -> BaseCodeAnalyzer:
        """Create a default analyzer (Java)."""
        return JavaCodeAnalyzer(AnalyzerConfig())
    
    @staticmethod
    def create_spring_boot_analyzer() -> BaseCodeAnalyzer:
        """Create an analyzer optimized for Spring Boot Java projects."""
        config = AnalyzerConfigBuilder().with_comment_removal(True).build()
        return JavaCodeAnalyzer(config)