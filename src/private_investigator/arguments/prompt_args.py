
from dataclasses import dataclass, field

@dataclass
class PromptArgs:
    surrogate: str = field(default='pre', metadata={
        'help': 'type of the surrogate model',
        'choices': ['pre', 'fine']
    })

    target: str = field(default='email', metadata={
        'help': 'target PII',
        'choices': ['email', 'phone', 'name']
    })

    prompt_len: int = field(default=1, metadata={
        'help': 'number of tokens in prompt',
    })

    from_scratch: bool = field(default=False, metadata={
        'help': 'generate prompt from scratch'
    })

