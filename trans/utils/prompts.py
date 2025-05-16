CUDA_CPP_TRANSLATE_ZERO_SHOT="""You are an expert in translating {obj.source} programs to {obj.target} programs. Given the {obj.source} program below, translate it to {obj.target}. Ensure that the {obj.target} program is exactly the same as the {obj.source} program input and output, and that the semantics of the original code are preserved.
Just generate the {obj.target} program and remove any unnecessary comments. Surround the generated {obj.target} program in [{obj.target}] and [/{obj.target}].  
### {obj.source} Program:
[{obj.source}]
{content}
[/{obj.source}]
### {obj.target} Version:
"""

CUDA_CPP_TRANSLATE_SYSTEM="""You are an expert in translating {obj.source} programs to {obj.target} programs. Given the {obj.source} program by User, translate it to {obj.target}. Ensure that the {obj.target} program is exactly the same as the {obj.source} program input and output, and that the semantics of the original code are preserved.
Just generate the {obj.target} program and remove any unnecessary comments. Surround the generated {obj.target} program in [{obj.target}] and [/{obj.target}].
"""

CUDA_CPP_TRANSLATE_USER="""
### {obj.source} Program:
[{obj.source}]
{content}
[/{obj.source}]
### {obj.target} Version:
"""

CUDA_CPP_TRANSLATE_TRAIN_SYSTEM="""You are an expert in translating {obj.source} programs to {obj.target} programs. Given the {obj.source} program by User, translate it to {obj.target}. Ensure that the {obj.target} program is exactly the same as the {obj.source} program input and output, and that the semantics of the original code are preserved.
Just generate the {obj.target} program and remove any unnecessary comments.
"""
