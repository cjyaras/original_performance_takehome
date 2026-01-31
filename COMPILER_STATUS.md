# Compiler Infrastructure Status

This document summarizes the compiler backend built for optimizing the kernel.

## Overview

A modular compiler was built in the `compiler/` directory that generates optimized instructions for the simulated VLIW SIMD CPU. The approach uses SSA-based IR with instruction scheduling and register allocation.

## Architecture

```
Kernel Spec → SSA IR → Scheduling → Register Allocation → Code Generation → Machine Instructions
```

## Modules

| File | Purpose | Status |
|------|---------|--------|
| `compiler/ssa.py` | SSA IR with Values, Constants, Instructions, IRBuilder | Complete |
| `compiler/scheduler.py` | Dependency graph + list scheduling for VLIW | Complete (but slow for large IR) |
| `compiler/regalloc.py` | Linear scan register allocation with reuse | Complete |
| `compiler/vectorize.py` | Framework for scalar → VALU conversion | Scaffolded, not implemented |
| `compiler/codegen.py` | Emit Machine instruction format | Complete |
| `compiler/kernel.py` | High-level kernel builder using all passes | Complete |

## Current Performance

| Configuration | Cycles | Notes |
|---------------|--------|-------|
| Baseline (original) | 147,734 | One instruction per cycle, no VLIW packing |
| Compiler (no scheduling) | 204,816 | Correct but slower (more const loads) |
| Compiler (with scheduling, small kernels) | ~95 for batch=8,rounds=1 | Works but scheduler too slow for full size |

**Full benchmark (batch=256, rounds=16)**: Scheduling is disabled because it's O(n²) and takes too long for 200k+ instructions.

## Correctness

The compiler passes all correctness tests across various configurations:
- batch sizes: 4, 8, 16, 32
- rounds: 1, 2, 4, 8

Verified by comparing output against `reference_kernel2()`.

## Key Design Decisions

1. **Fully unrolled**: No loops - all 256×16 iterations are expanded into IR
2. **SSA form**: Each value defined exactly once, explicit data dependencies
3. **Memory dependencies**: Cross-round dependencies tracked (round N+1 loads depend on round N stores for same batch element)
4. **Register reuse**: Linear scan allocation reclaims registers when values go dead

## TODO: Optimizations Needed for Speedup

### 1. Vectorization (Highest Priority)
- Use VALU to process 8 batch elements at once
- Reduces IR from 200k to ~25k instructions
- Makes scheduling feasible
- **Challenge**: Gather/scatter for tree node loads (indices differ per lane)

### 2. Fast Scheduling
Current scheduler is O(n²), too slow for large IR. Options:
- **Chunk-based**: Schedule each iteration independently, then concatenate
- **Greedy packing**: Simpler algorithm without full dependency graph
- **Hierarchical**: Schedule at round level, then batch level

### 3. Constant Hoisting
Currently each iteration loads its own constants. Should:
- Load hash constants once at start
- Reuse batch index constants across rounds
- Pre-compute common values

### 4. Instruction-Level Parallelism
The VLIW machine supports per cycle:
- 12 ALU ops
- 6 VALU ops
- 2 loads
- 2 stores
- 1 flow op

Current scheduling packs independent ops but could be more aggressive.

## How to Use the Compiler

```python
from compiler.kernel import KernelConfig, CompiledKernel

config = KernelConfig(
    forest_height=10,
    n_nodes=2047,
    batch_size=256,
    rounds=16,
    debug=True  # Print IR stats
)

kernel = CompiledKernel(config)
instructions = kernel.compile()  # Returns list of Machine instruction dicts
```

## Testing

```bash
# Run correctness tests (uses original KernelBuilder, not compiler)
python perf_takehome.py

# Run benchmark tests
python tests/submission_tests.py

# Test compiler directly
python -c "
from compiler.kernel import KernelConfig, CompiledKernel
config = KernelConfig(forest_height=4, n_nodes=31, batch_size=8, rounds=2)
kernel = CompiledKernel(config)
instrs = kernel.compile()
print(f'Generated {len(instrs)} cycles')
"
```

## Files Modified

- `CLAUDE.md` - Added instruction not to modify tests/
- `compiler/` - New directory with all compiler modules

## Next Steps for Continuation

1. **Implement vectorization** in `compiler/vectorize.py`:
   - Generate vector IR directly (not by converting scalar)
   - Handle gather loads for tree nodes
   - Use vbroadcast for constants

2. **Optimize scheduler** for large IR:
   - Add chunk-based scheduling option
   - Or implement simpler greedy packing

3. **Integrate with perf_takehome.py**:
   - Replace `KernelBuilder.build_kernel()` with compiler
   - Ensure it passes `tests/submission_tests.py`
