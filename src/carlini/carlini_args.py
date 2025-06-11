
from dataclasses import dataclass, field

@dataclass
class CarliniArgs:
    target: str = field(default='email', metadata={
        'help': 'target PII',
        'choices': ['email', 'phone', 'name']
    })

    N: int = field(default=200000, metadata={
        'help': 'Number of samples to generate'
    })

    n: int = field(default=100, metadata={
        'help': 'Number of samples to select'
    })

    batch_size: int = field(default=100, metadata={
        'help': 'Batch size for generation'
    })

    method: str = field(default='topk', metadata={
        'help': 'Generation method',
        'choices': ['topk', 'internet', 'temperature']
    })

    temp: float = field(default=1, metadata={
        'help': 'temperature value used during generation'
    })

    wet_file: str = field(default='../dataset/commoncrawl.warc.wet', metadata={
        'help': 'path to internet crawling text'
    })

    model_path: str = field(default='weights/gpt2-enron', metadata={
        'help': 'path to target model'
    })

    seed: int = field(default=0, metadata={
        'help': 'seed'
    })

    result_path: str = field(default='carlini_result', metadata={
        'help': 'folder to store results'
    })