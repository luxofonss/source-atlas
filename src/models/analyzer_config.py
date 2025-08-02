from dataclasses import dataclass, field
from typing import Set, Dict, List
from models.domain_models import ClassType

@dataclass(frozen=True)
class AnalyzerConfig:
    """Configuration for the code analyzer, supporting multiple languages."""
    remove_comments: bool = True
    builtin_types: Set[str] = field(default_factory=lambda: {
        'byte', 'short', 'int', 'long', 'float', 'double', 'boolean', 'char', 'void',
        'String', 'Object', 'Class', 'Integer', 'Long', 'Double', 'Float', 'Boolean',
        'Character', 'Byte', 'Short', 'BigDecimal', 'BigInteger', 'Date', 'LocalDate',
        'LocalDateTime', 'LocalTime', 'Instant', 'Duration', 'Period', 'List', 'Set',
        'Map', 'Collection', 'Optional', 'Stream', 'Future', 'CompletableFuture'
    })
    springboot_annotations: Dict[str, ClassType] = field(default_factory=lambda: {
        '@Configuration': ClassType.CONFIGURATION,
        '@EnableAutoConfiguration': ClassType.CONFIGURATION,
        '@SpringBootApplication': ClassType.CONFIGURATION,
        '@Service': ClassType.SERVICE,
        '@Controller': ClassType.CONTROLLER,
        '@RestController': ClassType.CONTROLLER,
        '@Repository': ClassType.REPOSITORY,
        '@Entity': ClassType.ENTITY,
        '@Table': ClassType.ENTITY,
        '@Document': ClassType.ENTITY,
        '@Filter': ClassType.FILTER,
        '@Component': ClassType.COMPONENT,
        '@ControllerAdvice': ClassType.CONTROLLER_ADVICE,
        '@RestControllerAdvice': ClassType.CONTROLLER_ADVICE,
        '@Bean': ClassType.BEAN,
    })
    job_patterns: List[str] = field(default_factory=lambda: [
        r'@Scheduled\s*\(',
        r'@Job\s*\(',
        r'@EnableScheduling',
        r'JobLauncher',
        r'QuartzJob',
        r'@Async\s*\(',
        r'TaskExecutor',
        r'ThreadPoolTaskExecutor',
        r'@EnableAsync'
    ])
    publisher_patterns: List[str] = field(default_factory=lambda: [
        r'ApplicationEventPublisher',
        r'@EventListener',
        r'@RabbitTemplate',
        r'@KafkaTemplate',
        r'RedisTemplate',
        r'JmsTemplate',
        r'@TransactionalEventListener',
        r'@Publish',
        r'MessageProducer',
        r'\.publish\s*\(',
        r'\.send\s*\(',
        r'\.convertAndSend\s*\('
    ])
    listener_patterns: List[str] = field(default_factory=lambda: [
        r'@EventListener\s*\(',
        r'@RabbitListener\s*\(',
        r'@KafkaListener\s*\(',
        r'@JmsListener\s*\(',
        r'@StreamListener\s*\(',
        r'MessageListener',
        r'@Subscribe',
        r'@Handler',
        r'ApplicationListener',
        r'@Component.*Listener'
    ])
    # Add language-specific configurations as needed
    language_specific_configs: Dict[str, Dict] = field(default_factory=dict)