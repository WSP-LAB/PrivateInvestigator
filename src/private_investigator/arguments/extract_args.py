
from dataclasses import dataclass, field

@dataclass
class ExtractArgs:
    c: float = field(default=0.25, metadata={
        'help': 'balance constant for exploration term of prompt score'
    })

    n: int = field(default=100, metadata={
        'help': 'number of attack attempts'
    })

    num_prompts: int = field(default=20, metadata={
        'help': 'number of generated prompts'
    })

    result_path: str = field(default=None, metadata={
        'help': 'folder to store results'
    })

