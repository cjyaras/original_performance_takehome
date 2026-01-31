"""
Kernel Builder

Builds the tree traversal kernel using the compiler infrastructure.
This generates SSA IR, schedules it, allocates registers, and emits code.
"""

from typing import Optional
from dataclasses import dataclass

from .ssa import IRBuilder, Value, Constant, Instruction, Engine, print_ir
from .scheduler import schedule, print_schedule
from .regalloc import allocate, simple_allocate, Allocation, VLEN, SCRATCH_SIZE
from .codegen import generate, CodeGenerator


# Hash stages from problem.py
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]


@dataclass
class KernelConfig:
    """Configuration for kernel generation"""
    forest_height: int
    n_nodes: int
    batch_size: int
    rounds: int
    use_vectorization: bool = False
    debug: bool = False


class KernelIRBuilder:
    """
    Builds SSA IR for the tree traversal kernel.
    """

    def __init__(self, config: KernelConfig):
        self.config = config
        self.builder = IRBuilder()

        # Pre-allocated values for init variables
        self.init_vars: dict[str, Value] = {}

    def build(self) -> list[Instruction]:
        """Build the kernel IR"""
        # Setup: load initial variables
        self._build_setup()

        # Main loop body (fully unrolled)
        for round_idx in range(self.config.rounds):
            for batch_idx in range(self.config.batch_size):
                self._build_iteration(round_idx, batch_idx)

        return self.builder.instructions

    def _build_setup(self):
        """Build setup code to load init variables from memory"""
        init_var_names = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p"
        ]

        for i, name in enumerate(init_var_names):
            # Load address constant
            addr = self.builder.const_load(i, name=f"addr_{name}")
            # Load value from memory
            val = self.builder.load(addr, name=name)
            self.init_vars[name] = val

    def _build_iteration(self, round_idx: int, batch_idx: int):
        """Build one iteration of the loop (one batch element, one round)"""
        meta = {"round_index": round_idx, "batch_index": batch_idx}

        # idx = mem[inp_indices_p + batch_idx]
        batch_const = self.builder.const_load(batch_idx)
        idx_addr = self.builder.add(self.init_vars["inp_indices_p"], batch_const, **meta)
        idx = self.builder.load(idx_addr, name="idx", **meta)

        # val = mem[inp_values_p + batch_idx]
        val_addr = self.builder.add(self.init_vars["inp_values_p"], batch_const, **meta)
        val = self.builder.load(val_addr, name="val", **meta)

        # node_val = mem[forest_values_p + idx]
        node_addr = self.builder.add(self.init_vars["forest_values_p"], idx, **meta)
        node_val = self.builder.load(node_addr, name="node_val", **meta)

        # val = myhash(val ^ node_val)
        xored = self.builder.xor(val, node_val, **meta)
        hashed = self._build_hash(xored, round_idx, batch_idx)

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        two = self.builder.const_load(2)
        zero = self.builder.const_load(0)
        one = self.builder.const_load(1)

        mod_result = self.builder.mod(hashed, two, **meta)
        is_even = self.builder.eq(mod_result, zero, **meta)

        # offset = 1 if even else 2
        offset = self.builder.select(is_even, one, two, **meta)

        # new_idx = 2*idx + offset
        idx_times_2 = self.builder.mul(idx, two, **meta)
        new_idx = self.builder.add(idx_times_2, offset, **meta)

        # idx = 0 if idx >= n_nodes else idx
        is_valid = self.builder.lt(new_idx, self.init_vars["n_nodes"], **meta)
        final_idx = self.builder.select(is_valid, new_idx, zero, **meta)

        # Store results
        # mem[inp_indices_p + batch_idx] = final_idx
        self.builder.store(idx_addr, final_idx, **meta)
        # mem[inp_values_p + batch_idx] = hashed
        self.builder.store(val_addr, hashed, **meta)

    def _build_hash(self, val: Value, round_idx: int, batch_idx: int) -> Value:
        """Build the hash computation (6 stages)"""
        meta = {"round_index": round_idx, "batch_index": batch_idx}

        current = val
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            const1 = self.builder.const_load(val1)
            const3 = self.builder.const_load(val3)

            # tmp1 = current op1 val1
            if op1 == "+":
                tmp1 = self.builder.add(current, const1, **meta)
            else:  # op1 == "^"
                tmp1 = self.builder.xor(current, const1, **meta)

            # tmp2 = current op3 val3
            if op3 == "<<":
                tmp2 = self.builder.shl(current, const3, **meta)
            else:  # op3 == ">>"
                tmp2 = self.builder.shr(current, const3, **meta)

            # current = tmp1 op2 tmp2
            if op2 == "+":
                current = self.builder.add(tmp1, tmp2, **meta)
            else:  # op2 == "^"
                current = self.builder.xor(tmp1, tmp2, **meta)

        return current


class CompiledKernel:
    """
    A fully compiled kernel ready to be executed by the Machine.
    """

    def __init__(self, config: KernelConfig):
        self.config = config
        self.instructions: list[dict] = []
        self.allocation: Optional[Allocation] = None
        self.debug_info: dict = {}

    def compile(self) -> list[dict]:
        """
        Compile the kernel through all passes.
        Returns list of Machine instruction dicts.
        """
        # 1. Build IR
        ir_builder = KernelIRBuilder(self.config)
        ir = ir_builder.build()

        if self.config.debug:
            print(f"Generated {len(ir)} IR instructions")
            print_ir(ir[:50])  # Print first 50

        # 2. Schedule (skip for large kernels - too slow)
        from .scheduler import ScheduledBundle
        if len(ir) > 10000:
            # For large kernels, just emit one instruction per cycle
            bundles = [ScheduledBundle(cycle=i, instructions=[instr]) for i, instr in enumerate(ir)]
            if self.config.debug:
                print(f"Skipped scheduling (too many instructions)")
        else:
            bundles = schedule(ir)
            if self.config.debug:
                print(f"Scheduled into {len(bundles)} cycles")
                print_schedule(bundles[:10])

        # 3. Allocate registers (with reuse)
        self.allocation = allocate(bundles)

        if self.config.debug:
            print(f"Allocated {self.allocation.max_addr} scratch words")

        # 4. Generate code
        self.instructions = generate(bundles, self.allocation)

        # 5. Add pause instructions for debugging sync
        self.instructions.insert(0, {"flow": [("pause",)]})
        self.instructions.append({"flow": [("pause",)]})

        return self.instructions

    def get_debug_info(self) -> dict:
        """Get debug info for the Machine"""
        scratch_map = {}
        if self.allocation:
            for val, addr in self.allocation.value_to_addr.items():
                size = VLEN if val.is_vector else 1
                name = val.name or f"v{val.id}"
                scratch_map[addr] = (name, size)
        return {"scratch_map": scratch_map}


def build_kernel(
    forest_height: int,
    n_nodes: int,
    batch_size: int,
    rounds: int,
    debug: bool = False
) -> tuple[list[dict], dict]:
    """
    Build the kernel using the compiler infrastructure.

    Returns:
        (instructions, debug_info) tuple
    """
    config = KernelConfig(
        forest_height=forest_height,
        n_nodes=n_nodes,
        batch_size=batch_size,
        rounds=rounds,
        debug=debug
    )

    kernel = CompiledKernel(config)
    instructions = kernel.compile()
    debug_info = kernel.get_debug_info()

    return instructions, debug_info
