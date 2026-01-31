"""
Register Allocator

Maps SSA virtual registers to physical scratch addresses using linear scan allocation.
Handles both scalar (1 word) and vector (8 word) values.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from .ssa import Instruction, Value, Constant, Engine
from .scheduler import ScheduledBundle


VLEN = 8  # Vector length
SCRATCH_SIZE = 1536  # Total scratch space available


@dataclass
class LiveRange:
    """The live range of a value (from definition to last use)"""
    value: Value
    start: int  # Cycle where value is defined
    end: int    # Last cycle where value is used
    size: int   # 1 for scalar, VLEN for vector

    def overlaps(self, other: 'LiveRange') -> bool:
        """Check if two live ranges overlap"""
        return not (self.end < other.start or other.end < self.start)


@dataclass
class Allocation:
    """Maps values to scratch addresses"""
    value_to_addr: dict[Value, int] = field(default_factory=dict)
    const_to_addr: dict[int, int] = field(default_factory=dict)  # For constant values
    next_addr: int = 0
    max_addr: int = 0

    def alloc(self, value: Value) -> int:
        """Allocate scratch space for a value"""
        size = VLEN if value.is_vector else 1
        addr = self.next_addr
        self.value_to_addr[value] = addr
        self.next_addr += size
        self.max_addr = max(self.max_addr, self.next_addr)
        return addr

    def get(self, value: Value) -> int:
        """Get the scratch address for a value"""
        return self.value_to_addr[value]

    def alloc_const(self, const_val: int) -> int:
        """Allocate scratch space for a constant"""
        if const_val not in self.const_to_addr:
            addr = self.next_addr
            self.const_to_addr[const_val] = addr
            self.next_addr += 1
            self.max_addr = max(self.max_addr, self.next_addr)
        return self.const_to_addr[const_val]


class LivenessAnalyzer:
    """Compute live ranges for all values in scheduled code"""

    def __init__(self, bundles: list[ScheduledBundle]):
        self.bundles = bundles
        self.live_ranges: dict[Value, LiveRange] = {}
        self._analyze()

    def _analyze(self):
        """Compute live ranges"""
        # Map from value to (def_cycle, last_use_cycle)
        def_cycle: dict[Value, int] = {}
        last_use: dict[Value, int] = {}

        for bundle in self.bundles:
            cycle = bundle.cycle
            for instr in bundle.instructions:
                # Record definition
                if instr.result is not None:
                    def_cycle[instr.result] = cycle

                # Record uses
                for op in instr.operands:
                    if isinstance(op, Value):
                        last_use[op] = cycle

        # Build live ranges
        for value, start in def_cycle.items():
            end = last_use.get(value, start)  # If never used, live range is just definition
            size = VLEN if value.is_vector else 1
            self.live_ranges[value] = LiveRange(value=value, start=start, end=end, size=size)


class LinearScanAllocator:
    """
    Linear scan register allocation.

    Allocates scratch addresses to values, reusing space when possible.
    Values with non-overlapping live ranges can share the same address.
    """

    def __init__(self, bundles: list[ScheduledBundle]):
        self.bundles = bundles
        self.liveness = LivenessAnalyzer(bundles)

    def allocate(self) -> Allocation:
        """
        Perform linear scan allocation with register reuse.
        Returns mapping from Values to scratch addresses.
        """
        allocation = Allocation()

        # Sort live ranges by start point
        ranges = sorted(self.liveness.live_ranges.values(), key=lambda r: r.start)

        # Track active intervals: (addr, size, end_cycle, value)
        active: list[tuple[int, int, int, Value]] = []

        # Free list: list of (addr, size) tuples, sorted by size then addr
        free_scalars: list[int] = []  # Free scalar addresses
        free_vectors: list[int] = []  # Free vector addresses (8-word aligned)

        for lr in ranges:
            # Expire old intervals that ended before this one starts
            new_active = []
            for addr, size, end, val in active:
                if end < lr.start:
                    # This interval has expired, reclaim its space
                    if size == 1:
                        free_scalars.append(addr)
                    else:
                        free_vectors.append(addr)
                else:
                    new_active.append((addr, size, end, val))
            active = new_active

            # Try to reuse a free address
            size = VLEN if lr.value.is_vector else 1
            addr = None

            if size == 1 and free_scalars:
                addr = free_scalars.pop()
            elif size == VLEN and free_vectors:
                addr = free_vectors.pop()

            if addr is not None:
                allocation.value_to_addr[lr.value] = addr
            else:
                # Allocate new space
                addr = allocation.alloc(lr.value)

            active.append((addr, size, lr.end, lr.value))

        return allocation


def allocate(bundles: list[ScheduledBundle]) -> Allocation:
    """Convenience function to allocate registers"""
    allocator = LinearScanAllocator(bundles)
    return allocator.allocate()


class SimpleAllocator:
    """
    Simpler allocation strategy: just give each value a unique address.
    Doesn't reuse space but is correct and simple.
    """

    def __init__(self, bundles: list[ScheduledBundle]):
        self.bundles = bundles

    def allocate(self) -> Allocation:
        allocation = Allocation()

        # Collect all values that need allocation
        for bundle in self.bundles:
            for instr in bundle.instructions:
                if instr.result is not None and instr.result not in allocation.value_to_addr:
                    allocation.alloc(instr.result)

        return allocation


def simple_allocate(bundles: list[ScheduledBundle]) -> Allocation:
    """Simple allocation without reuse"""
    allocator = SimpleAllocator(bundles)
    return allocator.allocate()
