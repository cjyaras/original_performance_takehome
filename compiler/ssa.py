"""
SSA (Static Single Assignment) Intermediate Representation

This module defines the IR used by the compiler. Each value is defined exactly once,
making data dependencies explicit and enabling optimization passes.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum, auto


class Engine(Enum):
    """Which execution engine handles this operation"""
    ALU = auto()    # Scalar arithmetic
    VALU = auto()   # Vector arithmetic (VLEN=8)
    LOAD = auto()   # Memory loads
    STORE = auto()  # Memory stores
    FLOW = auto()   # Control flow, select


@dataclass
class Value:
    """
    A virtual register in SSA form. Each Value is defined exactly once.
    """
    id: int
    name: Optional[str] = None  # For debugging
    is_vector: bool = False     # True if this is a vector (8 elements)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.id == other.id
        return False

    def __repr__(self):
        prefix = "v" if self.is_vector else "s"
        if self.name:
            return f"{prefix}{self.id}:{self.name}"
        return f"{prefix}{self.id}"


@dataclass
class Constant:
    """An immediate constant value"""
    value: int

    def __hash__(self):
        return hash(("const", self.value))

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"${self.value}"


Operand = Union[Value, Constant]


@dataclass
class Instruction:
    """
    An SSA instruction. Result is None for instructions with side effects (stores).
    """
    opcode: str
    operands: list[Operand]
    result: Optional[Value] = None
    engine: Engine = Engine.ALU

    # Metadata for optimization passes
    batch_index: Optional[int] = None  # Which batch element (0-255)
    round_index: Optional[int] = None  # Which round (0-15)

    def __repr__(self):
        ops_str = ", ".join(str(op) for op in self.operands)
        if self.result:
            return f"{self.result} = {self.opcode}({ops_str})"
        return f"{self.opcode}({ops_str})"

    def uses(self) -> list[Value]:
        """Return all Values this instruction reads"""
        return [op for op in self.operands if isinstance(op, Value)]

    def defines(self) -> Optional[Value]:
        """Return the Value this instruction defines, if any"""
        return self.result


class IRBuilder:
    """
    Builder for constructing SSA IR. Handles value numbering and instruction creation.
    """

    def __init__(self):
        self.instructions: list[Instruction] = []
        self.next_id = 0
        self.constants: dict[int, Value] = {}  # Cache for constant values

    def new_value(self, name: Optional[str] = None, is_vector: bool = False) -> Value:
        """Allocate a new virtual register"""
        v = Value(id=self.next_id, name=name, is_vector=is_vector)
        self.next_id += 1
        return v

    def const(self, value: int) -> Constant:
        """Create a constant operand"""
        return Constant(value)

    def _add_instr(self, opcode: str, operands: list[Operand], engine: Engine,
                   result_name: Optional[str] = None, is_vector: bool = False,
                   batch_index: Optional[int] = None, round_index: Optional[int] = None) -> Value:
        """Add an instruction that produces a value"""
        result = self.new_value(result_name, is_vector=is_vector)
        instr = Instruction(
            opcode=opcode,
            operands=operands,
            result=result,
            engine=engine,
            batch_index=batch_index,
            round_index=round_index
        )
        self.instructions.append(instr)
        return result

    def _add_void_instr(self, opcode: str, operands: list[Operand], engine: Engine,
                        batch_index: Optional[int] = None, round_index: Optional[int] = None):
        """Add an instruction with no result (e.g., store)"""
        instr = Instruction(
            opcode=opcode,
            operands=operands,
            result=None,
            engine=engine,
            batch_index=batch_index,
            round_index=round_index
        )
        self.instructions.append(instr)

    # === ALU Operations ===

    def add(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("+", [a, b], Engine.ALU, name, **meta)

    def sub(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("-", [a, b], Engine.ALU, name, **meta)

    def mul(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("*", [a, b], Engine.ALU, name, **meta)

    def xor(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("^", [a, b], Engine.ALU, name, **meta)

    def and_(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("&", [a, b], Engine.ALU, name, **meta)

    def or_(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("|", [a, b], Engine.ALU, name, **meta)

    def shl(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("<<", [a, b], Engine.ALU, name, **meta)

    def shr(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr(">>", [a, b], Engine.ALU, name, **meta)

    def mod(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("%", [a, b], Engine.ALU, name, **meta)

    def lt(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("<", [a, b], Engine.ALU, name, **meta)

    def eq(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("==", [a, b], Engine.ALU, name, **meta)

    # === VALU Operations (Vector) ===

    def vadd(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v+", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vsub(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v-", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vmul(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v*", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vxor(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v^", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vand(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v&", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vor(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v|", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vshl(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v<<", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vshr(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v>>", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vmod(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v%", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vlt(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v<", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def veq(self, a: Operand, b: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("v==", [a, b], Engine.VALU, name, is_vector=True, **meta)

    def vbroadcast(self, scalar: Operand, name: str = None, **meta) -> Value:
        return self._add_instr("vbroadcast", [scalar], Engine.VALU, name, is_vector=True, **meta)

    # === Load Operations ===

    def load(self, addr: Operand, name: str = None, **meta) -> Value:
        """Load from mem[scratch[addr]]"""
        return self._add_instr("load", [addr], Engine.LOAD, name, **meta)

    def vload(self, addr: Operand, name: str = None, **meta) -> Value:
        """Load 8 consecutive elements from mem[scratch[addr]]"""
        return self._add_instr("vload", [addr], Engine.LOAD, name, is_vector=True, **meta)

    def const_load(self, value: int, name: str = None, **meta) -> Value:
        """Load an immediate constant into scratch"""
        return self._add_instr("const", [Constant(value)], Engine.LOAD, name, **meta)

    # === Store Operations ===

    def store(self, addr: Operand, value: Operand, **meta):
        """Store scratch[value] to mem[scratch[addr]]"""
        self._add_void_instr("store", [addr, value], Engine.STORE, **meta)

    def vstore(self, addr: Operand, value: Operand, **meta):
        """Store 8 elements from scratch to mem[scratch[addr]]"""
        self._add_void_instr("vstore", [addr, value], Engine.STORE, **meta)

    # === Flow Operations ===

    def select(self, cond: Operand, true_val: Operand, false_val: Operand,
               name: str = None, **meta) -> Value:
        """Select true_val if cond != 0, else false_val"""
        return self._add_instr("select", [cond, true_val, false_val], Engine.FLOW, name, **meta)

    def vselect(self, cond: Operand, true_val: Operand, false_val: Operand,
                name: str = None, **meta) -> Value:
        """Vector select"""
        return self._add_instr("vselect", [cond, true_val, false_val], Engine.FLOW, name, is_vector=True, **meta)


def print_ir(instructions: list[Instruction]):
    """Pretty print IR for debugging"""
    for i, instr in enumerate(instructions):
        print(f"{i:4d}: {instr}")
