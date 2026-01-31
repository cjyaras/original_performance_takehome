"""
Instruction Scheduler

Performs list scheduling to pack instructions into VLIW bundles while
respecting data dependencies and resource constraints.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from .ssa import Instruction, Value, Constant, Engine, Operand


# Resource limits per cycle (from problem.py)
SLOT_LIMITS = {
    Engine.ALU: 12,
    Engine.VALU: 6,
    Engine.LOAD: 2,
    Engine.STORE: 2,
    Engine.FLOW: 1,
}


@dataclass
class ScheduledBundle:
    """A VLIW instruction bundle - all ops execute in one cycle"""
    cycle: int
    instructions: list[Instruction] = field(default_factory=list)

    def slots_used(self) -> dict[Engine, int]:
        """Count how many slots of each engine are used"""
        counts = defaultdict(int)
        for instr in self.instructions:
            counts[instr.engine] += 1
        return counts

    def can_add(self, instr: Instruction) -> bool:
        """Check if we can add this instruction without exceeding slot limits"""
        used = self.slots_used()
        return used[instr.engine] < SLOT_LIMITS[instr.engine]

    def add(self, instr: Instruction):
        """Add instruction to this bundle"""
        self.instructions.append(instr)


class DependencyGraph:
    """
    Tracks data dependencies between instructions.
    Uses RAW (Read After Write) dependencies from SSA form.
    """

    def __init__(self, instructions: list[Instruction]):
        self.instructions = instructions
        self.n = len(instructions)

        # Map from Value -> instruction index that defines it
        self.def_map: dict[Value, int] = {}

        # Adjacency lists: predecessors[i] = instructions that must complete before i
        self.predecessors: dict[int, set[int]] = defaultdict(set)
        # successors[i] = instructions that depend on i
        self.successors: dict[int, set[int]] = defaultdict(set)

        self._build()

    def _build(self):
        """Build the dependency graph"""
        # First pass: record which instruction defines each value
        for i, instr in enumerate(self.instructions):
            if instr.result is not None:
                self.def_map[instr.result] = i

        # Second pass: add edges for RAW (Read After Write) dependencies only
        # This is sufficient since we're fully unrolling and generating code in order
        for i, instr in enumerate(self.instructions):
            for operand in instr.operands:
                if isinstance(operand, Value) and operand in self.def_map:
                    pred = self.def_map[operand]
                    if pred != i:  # No self-loops
                        self.predecessors[i].add(pred)
                        self.successors[pred].add(i)

        # Add memory ordering dependencies between rounds
        # Within a round, different batch elements access different addresses
        # Across rounds, the same batch element accesses the same addresses
        # So: round N+1's loads depend on round N's stores for the same batch element

        # Find stores grouped by (round, batch)
        stores_by_round_batch: dict[tuple, list[int]] = defaultdict(list)
        for i, instr in enumerate(self.instructions):
            if instr.engine == Engine.STORE and instr.round_index is not None:
                key = (instr.round_index, instr.batch_index)
                stores_by_round_batch[key].append(i)

        # Add dependencies: loads in round N+1 depend on stores in round N
        for i, instr in enumerate(self.instructions):
            if instr.engine == Engine.LOAD and instr.opcode != "const" and instr.round_index is not None:
                prev_round = instr.round_index - 1
                if prev_round >= 0:
                    key = (prev_round, instr.batch_index)
                    for store_idx in stores_by_round_batch.get(key, []):
                        self.predecessors[i].add(store_idx)
                        self.successors[store_idx].add(i)

    def get_roots(self) -> list[int]:
        """Get instructions with no dependencies (can execute first)"""
        return [i for i in range(self.n) if len(self.predecessors[i]) == 0]

    def depth(self) -> dict[int, int]:
        """
        Compute the critical path depth for each instruction.
        Used for prioritization in list scheduling.
        Uses iterative approach to handle large graphs.
        """
        depths = {}

        # Process in reverse topological order (nodes with no successors first)
        # Use iterative approach with explicit stack to avoid recursion limits
        in_progress = set()

        for start in range(self.n):
            if start in depths:
                continue

            stack = [(start, False)]  # (node, processed)

            while stack:
                node, processed = stack.pop()

                if processed:
                    # All successors have been processed, compute depth
                    if not self.successors[node]:
                        depths[node] = 0
                    else:
                        depths[node] = 1 + max(depths[s] for s in self.successors[node])
                    in_progress.discard(node)
                elif node in depths:
                    # Already computed
                    continue
                elif node in in_progress:
                    # Cycle detected - shouldn't happen in valid SSA
                    depths[node] = 0
                else:
                    # Mark as in progress and push back with processed=True
                    in_progress.add(node)
                    stack.append((node, True))

                    # Push all unprocessed successors
                    for succ in self.successors[node]:
                        if succ not in depths and succ not in in_progress:
                            stack.append((succ, False))

        return depths


class ListScheduler:
    """
    List scheduling algorithm for VLIW instruction bundles.

    Schedules instructions as early as possible while respecting:
    1. Data dependencies
    2. Resource constraints (slot limits per engine)
    """

    def __init__(self, instructions: list[Instruction]):
        self.instructions = instructions
        self.graph = DependencyGraph(instructions)
        self.depths = self.graph.depth()

    def schedule(self) -> list[ScheduledBundle]:
        """
        Perform list scheduling.
        Returns a list of instruction bundles (one per cycle).
        """
        n = len(self.instructions)
        if n == 0:
            return []

        # Track when each instruction is scheduled
        scheduled_cycle: dict[int, int] = {}

        # Track which instructions are ready (all predecessors scheduled)
        ready: set[int] = set(self.graph.get_roots())

        # Count of unscheduled predecessors for each instruction
        pred_count = {i: len(self.graph.predecessors[i]) for i in range(n)}

        bundles: list[ScheduledBundle] = []
        current_cycle = 0

        while len(scheduled_cycle) < n:
            if not ready:
                # No ready instructions - should not happen if graph is valid
                raise RuntimeError("Deadlock in scheduling - no ready instructions")

            bundle = ScheduledBundle(cycle=current_cycle)

            # Sort ready instructions by priority (higher depth = higher priority)
            # This prioritizes instructions on the critical path
            ready_list = sorted(ready, key=lambda i: -self.depths[i])

            # Try to fill the bundle
            scheduled_this_cycle = []
            for i in ready_list:
                instr = self.instructions[i]
                if bundle.can_add(instr):
                    bundle.add(instr)
                    scheduled_this_cycle.append(i)

            # Record scheduled instructions
            for i in scheduled_this_cycle:
                ready.remove(i)
                scheduled_cycle[i] = current_cycle

                # Update successors
                for succ in self.graph.successors[i]:
                    pred_count[succ] -= 1
                    if pred_count[succ] == 0:
                        ready.add(succ)

            if bundle.instructions:
                bundles.append(bundle)

            current_cycle += 1

        return bundles


def schedule(instructions: list[Instruction]) -> list[ScheduledBundle]:
    """Convenience function to schedule a list of instructions"""
    scheduler = ListScheduler(instructions)
    return scheduler.schedule()


def print_schedule(bundles: list[ScheduledBundle]):
    """Pretty print scheduled bundles"""
    for bundle in bundles:
        print(f"Cycle {bundle.cycle}:")
        slots = bundle.slots_used()
        usage = ", ".join(f"{e.name}:{c}" for e, c in slots.items() if c > 0)
        print(f"  [{usage}]")
        for instr in bundle.instructions:
            print(f"    {instr}")
