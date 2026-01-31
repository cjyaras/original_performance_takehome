"""
Vectorization Pass

Converts groups of 8 scalar operations into single VALU (vector) operations.
This is possible because batch elements 0-7, 8-15, etc. are independent.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from .ssa import Instruction, Value, Constant, Engine, IRBuilder, Operand


VLEN = 8  # Number of elements per vector


@dataclass
class VectorGroup:
    """A group of VLEN scalar instructions that can be vectorized"""
    opcode: str
    round_index: int
    base_batch_index: int  # batch_index // VLEN * VLEN
    instructions: list[Instruction]  # Exactly VLEN instructions
    lane_to_instr: dict[int, Instruction] = field(default_factory=dict)


def get_vector_opcode(scalar_opcode: str) -> Optional[str]:
    """Convert scalar opcode to vector opcode"""
    # ALU ops that can be vectorized
    vectorizable = {
        "+": "+", "-": "-", "*": "*",
        "^": "^", "&": "&", "|": "|",
        "<<": "<<", ">>": ">>",
        "%": "%", "<": "<", "==": "==",
    }
    return vectorizable.get(scalar_opcode)


def can_vectorize(instr: Instruction) -> bool:
    """Check if an instruction can be part of a vector operation"""
    if instr.engine != Engine.ALU:
        return False
    if instr.batch_index is None:
        return False
    return get_vector_opcode(instr.opcode) is not None


class Vectorizer:
    """
    Converts scalar IR to vectorized IR.

    Strategy:
    1. Group instructions by (round, base_batch_index, opcode)
    2. For groups with exactly VLEN instructions, convert to VALU
    3. Handle operand dependencies by vectorizing in dependency order
    """

    def __init__(self, instructions: list[Instruction]):
        self.scalar_instructions = instructions
        self.vector_instructions: list[Instruction] = []

        # Map from scalar value to (vector_value, lane)
        self.scalar_to_vector: dict[Value, tuple[Value, int]] = {}

        # Map from (round, base_batch, lane) to vector value for operand lookup
        self.vector_values: dict[tuple[int, int, str], Value] = {}

        self.builder = IRBuilder()

    def vectorize(self) -> list[Instruction]:
        """
        Main vectorization pass.
        Returns vectorized instruction list.
        """
        # Group vectorizable instructions
        groups = self._find_vector_groups()

        # Process instructions in order, vectorizing where possible
        processed: set[int] = set()  # Indices of processed instructions

        # Build index map
        instr_to_idx = {id(instr): i for i, instr in enumerate(self.scalar_instructions)}

        for i, instr in enumerate(self.scalar_instructions):
            if i in processed:
                continue

            # Check if this instruction is part of a vector group
            group_key = self._get_group_key(instr)
            if group_key and group_key in groups:
                group = groups[group_key]
                if len(group.instructions) == VLEN:
                    # Vectorize this group
                    self._vectorize_group(group)
                    for gi in group.instructions:
                        processed.add(instr_to_idx[id(gi)])
                    continue

            # Not vectorizable - emit as scalar
            self._emit_scalar(instr)
            processed.add(i)

        return self.builder.instructions

    def _find_vector_groups(self) -> dict[tuple, VectorGroup]:
        """Find groups of VLEN instructions that can be vectorized together"""
        groups: dict[tuple, VectorGroup] = {}

        for instr in self.scalar_instructions:
            if not can_vectorize(instr):
                continue

            key = self._get_group_key(instr)
            if key is None:
                continue

            if key not in groups:
                round_idx, base_batch, opcode = key
                groups[key] = VectorGroup(
                    opcode=opcode,
                    round_index=round_idx,
                    base_batch_index=base_batch,
                    instructions=[],
                    lane_to_instr={}
                )

            lane = instr.batch_index % VLEN
            groups[key].instructions.append(instr)
            groups[key].lane_to_instr[lane] = instr

        return groups

    def _get_group_key(self, instr: Instruction) -> Optional[tuple]:
        """Get the group key for an instruction"""
        if not can_vectorize(instr):
            return None
        base_batch = (instr.batch_index // VLEN) * VLEN
        return (instr.round_index, base_batch, instr.opcode)

    def _vectorize_group(self, group: VectorGroup):
        """Convert a group of scalar instructions to a vector instruction"""
        # Get operands - need to handle both scalar and already-vectorized operands
        # For now, assume operands with same structure across lanes

        # Check that all lanes are present
        if len(group.lane_to_instr) != VLEN:
            # Fall back to scalar
            for instr in group.instructions:
                self._emit_scalar(instr)
            return

        # Get representative instruction (lane 0)
        rep = group.lane_to_instr[0]

        # Build vector operands
        vec_operands = []
        for op_idx, op in enumerate(rep.operands):
            vec_op = self._get_vector_operand(group, op_idx)
            vec_operands.append(vec_op)

        # Create vector instruction
        vec_opcode = group.opcode  # Same opcode, VALU engine handles it
        result = self.builder.new_value(
            name=f"v_{rep.result.name}" if rep.result and rep.result.name else None,
            is_vector=True
        )

        vec_instr = Instruction(
            opcode=vec_opcode,
            operands=vec_operands,
            result=result,
            engine=Engine.VALU,
            batch_index=group.base_batch_index,
            round_index=group.round_index
        )
        self.builder.instructions.append(vec_instr)

        # Record mapping from scalar results to vector lanes
        for lane, instr in group.lane_to_instr.items():
            if instr.result:
                self.scalar_to_vector[instr.result] = (result, lane)

    def _get_vector_operand(self, group: VectorGroup, op_idx: int) -> Operand:
        """
        Get the vector operand for a group at the given operand index.
        If all lanes use the same constant, broadcast it.
        If lanes use different scalar values, they should already be vectorized.
        """
        # Check if all lanes have the same operand
        operands = [group.lane_to_instr[lane].operands[op_idx] for lane in range(VLEN)]

        # All constants with same value?
        if all(isinstance(op, Constant) for op in operands):
            values = [op.value for op in operands]
            if len(set(values)) == 1:
                # Broadcast constant
                scalar = self.builder.const_load(values[0])
                return self.builder.vbroadcast(scalar)
            else:
                # Different constants per lane - need to load individually
                # For now, create a vector constant load
                # This is a simplification - in practice we'd need vload from memory
                scalar = self.builder.const_load(values[0])
                return self.builder.vbroadcast(scalar)

        # Check if operands are scalar values that were vectorized
        first_op = operands[0]
        if isinstance(first_op, Value) and first_op in self.scalar_to_vector:
            vec_val, _ = self.scalar_to_vector[first_op]
            return vec_val

        # Operands are scalar values not yet vectorized - need to handle
        # This happens when operands come from non-vectorizable instructions (like loads)
        # For now, return the first operand (this is a simplification)
        return first_op

    def _emit_scalar(self, instr: Instruction):
        """Emit a scalar instruction unchanged"""
        # Clone the instruction with new values from builder
        new_result = None
        if instr.result:
            new_result = self.builder.new_value(instr.result.name, instr.result.is_vector)

        new_instr = Instruction(
            opcode=instr.opcode,
            operands=list(instr.operands),  # Keep same operands for now
            result=new_result,
            engine=instr.engine,
            batch_index=instr.batch_index,
            round_index=instr.round_index
        )
        self.builder.instructions.append(new_instr)


def vectorize(instructions: list[Instruction]) -> list[Instruction]:
    """Convenience function to vectorize instructions"""
    vectorizer = Vectorizer(instructions)
    return vectorizer.vectorize()


class SimpleVectorizer:
    """
    A simpler vectorization approach:
    Generate vector IR directly from the kernel specification.

    Instead of converting scalar to vector, this builds vector IR from scratch
    by processing batch elements in groups of VLEN.
    """

    def __init__(self):
        self.builder = IRBuilder()

    def build_vectorized_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        hash_stages: list,
        init_vars: dict[str, Value],
    ) -> list[Instruction]:
        """
        Build vectorized kernel IR directly.

        This processes VLEN batch elements at a time using vector operations.
        """
        assert batch_size % VLEN == 0, f"batch_size must be multiple of VLEN ({VLEN})"

        n_vectors = batch_size // VLEN

        for round_idx in range(rounds):
            for vec_idx in range(n_vectors):
                base_batch = vec_idx * VLEN
                self._build_vector_iteration(
                    round_idx, base_batch, init_vars, n_nodes, hash_stages
                )

        return self.builder.instructions

    def _build_vector_iteration(
        self,
        round_idx: int,
        base_batch: int,
        init_vars: dict[str, Value],
        n_nodes: int,
        hash_stages: list,
    ):
        """Build one vector iteration (processes VLEN batch elements)"""
        meta = {"round_index": round_idx, "batch_index": base_batch}

        # Load indices: vload from inp_indices_p + base_batch
        idx_addr = self.builder.add(
            init_vars["inp_indices_p"],
            self.builder.const(base_batch),
            **meta
        )
        v_idx = self.builder.vload(idx_addr, name="v_idx", **meta)

        # Load values: vload from inp_values_p + base_batch
        val_addr = self.builder.add(
            init_vars["inp_values_p"],
            self.builder.const(base_batch),
            **meta
        )
        v_val = self.builder.vload(val_addr, name="v_val", **meta)

        # Load node values: need gather (not directly supported)
        # For now, this is a placeholder - real implementation needs scatter/gather
        # or falling back to scalar loads
        node_val_addr = self.builder.add(
            init_vars["forest_values_p"],
            v_idx,  # This won't work directly - need gather
            **meta
        )
        # v_node_val = self.builder.vload(node_val_addr, name="v_node_val", **meta)

        # ... rest of iteration would go here

        # This is incomplete - the gather/scatter problem makes full vectorization
        # non-trivial without additional support
