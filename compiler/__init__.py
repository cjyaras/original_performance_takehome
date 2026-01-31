"""
Compiler package for the performance take-home.

This package provides:
- SSA IR representation
- Instruction scheduling (list scheduling for VLIW)
- Register allocation (linear scan)
- Vectorization pass (scalar to VALU conversion)
- Code generation (emit Machine instruction format)
"""

from .ssa import (
    Value,
    Constant,
    Instruction,
    Engine,
    IRBuilder,
    print_ir,
)

from .scheduler import (
    ScheduledBundle,
    DependencyGraph,
    ListScheduler,
    schedule,
    print_schedule,
    SLOT_LIMITS,
)

from .regalloc import (
    LiveRange,
    Allocation,
    LivenessAnalyzer,
    LinearScanAllocator,
    allocate,
    simple_allocate,
    VLEN,
    SCRATCH_SIZE,
)

from .vectorize import (
    vectorize,
    Vectorizer,
    SimpleVectorizer,
)

from .codegen import (
    CodeGenerator,
    DirectCodeGenerator,
    generate,
)

__all__ = [
    # SSA
    "Value", "Constant", "Instruction", "Engine", "IRBuilder", "print_ir",
    # Scheduler
    "ScheduledBundle", "DependencyGraph", "ListScheduler", "schedule", "print_schedule", "SLOT_LIMITS",
    # Register Allocation
    "LiveRange", "Allocation", "LivenessAnalyzer", "LinearScanAllocator",
    "allocate", "simple_allocate", "VLEN", "SCRATCH_SIZE",
    # Vectorization
    "vectorize", "Vectorizer", "SimpleVectorizer",
    # Code Generation
    "CodeGenerator", "DirectCodeGenerator", "generate",
]
