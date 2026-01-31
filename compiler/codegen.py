"""
Code Generator

Converts scheduled IR with register allocation to the Machine instruction format.
Output format: list of dicts like {"alu": [...], "load": [...], ...}
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Any

from .ssa import Instruction, Value, Constant, Engine
from .scheduler import ScheduledBundle
from .regalloc import Allocation


def engine_to_str(engine: Engine) -> str:
    """Convert Engine enum to string for Machine format"""
    return {
        Engine.ALU: "alu",
        Engine.VALU: "valu",
        Engine.LOAD: "load",
        Engine.STORE: "store",
        Engine.FLOW: "flow",
    }[engine]


class CodeGenerator:
    """
    Generates Machine instruction format from scheduled IR.
    """

    def __init__(self, bundles: list[ScheduledBundle], allocation: Allocation):
        self.bundles = bundles
        self.allocation = allocation
        self.output: list[dict] = []

        # Track constants that need to be loaded
        self.const_load_instrs: list[dict] = []

    def generate(self) -> list[dict]:
        """
        Generate Machine instructions.
        Returns list of instruction dicts.
        """
        self.output = []

        # First, generate constant loads
        self._generate_const_loads()

        # Then generate main code
        for bundle in self.bundles:
            instr_dict = self._generate_bundle(bundle)
            if instr_dict:  # Don't emit empty bundles
                self.output.append(instr_dict)

        return self.const_load_instrs + self.output

    def _generate_const_loads(self):
        """Generate instructions to load constants into scratch"""
        for const_val, addr in self.allocation.const_to_addr.items():
            self.const_load_instrs.append({
                "load": [("const", addr, const_val)]
            })

    def _generate_bundle(self, bundle: ScheduledBundle) -> dict:
        """Generate a single VLIW instruction bundle"""
        result = defaultdict(list)

        for instr in bundle.instructions:
            engine_str = engine_to_str(instr.engine)
            slot = self._generate_slot(instr)
            if slot:
                result[engine_str].append(slot)

        return dict(result) if result else {}

    def _generate_slot(self, instr: Instruction) -> Optional[tuple]:
        """Generate a single instruction slot"""
        if instr.engine == Engine.ALU:
            return self._gen_alu(instr)
        elif instr.engine == Engine.VALU:
            return self._gen_valu(instr)
        elif instr.engine == Engine.LOAD:
            return self._gen_load(instr)
        elif instr.engine == Engine.STORE:
            return self._gen_store(instr)
        elif instr.engine == Engine.FLOW:
            return self._gen_flow(instr)
        return None

    def _resolve_operand(self, op) -> int:
        """Convert an operand to a scratch address"""
        if isinstance(op, Value):
            return self.allocation.get(op)
        elif isinstance(op, Constant):
            return self.allocation.alloc_const(op.value)
        else:
            raise ValueError(f"Unknown operand type: {type(op)}")

    def _gen_alu(self, instr: Instruction) -> tuple:
        """Generate ALU instruction: (op, dest, src1, src2)"""
        dest = self.allocation.get(instr.result)
        src1 = self._resolve_operand(instr.operands[0])
        src2 = self._resolve_operand(instr.operands[1])
        return (instr.opcode, dest, src1, src2)

    def _gen_valu(self, instr: Instruction) -> tuple:
        """Generate VALU instruction"""
        if instr.opcode == "vbroadcast":
            dest = self.allocation.get(instr.result)
            src = self._resolve_operand(instr.operands[0])
            return ("vbroadcast", dest, src)
        else:
            # Binary vector op: (op, dest, src1, src2)
            dest = self.allocation.get(instr.result)
            src1 = self._resolve_operand(instr.operands[0])
            src2 = self._resolve_operand(instr.operands[1])
            return (instr.opcode, dest, src1, src2)

    def _gen_load(self, instr: Instruction) -> tuple:
        """Generate load instruction"""
        if instr.opcode == "const":
            dest = self.allocation.get(instr.result)
            value = instr.operands[0].value
            return ("const", dest, value)
        elif instr.opcode == "load":
            dest = self.allocation.get(instr.result)
            addr = self._resolve_operand(instr.operands[0])
            return ("load", dest, addr)
        elif instr.opcode == "vload":
            dest = self.allocation.get(instr.result)
            addr = self._resolve_operand(instr.operands[0])
            return ("vload", dest, addr)
        else:
            raise ValueError(f"Unknown load opcode: {instr.opcode}")

    def _gen_store(self, instr: Instruction) -> tuple:
        """Generate store instruction"""
        if instr.opcode == "store":
            addr = self._resolve_operand(instr.operands[0])
            src = self._resolve_operand(instr.operands[1])
            return ("store", addr, src)
        elif instr.opcode == "vstore":
            addr = self._resolve_operand(instr.operands[0])
            src = self._resolve_operand(instr.operands[1])
            return ("vstore", addr, src)
        else:
            raise ValueError(f"Unknown store opcode: {instr.opcode}")

    def _gen_flow(self, instr: Instruction) -> tuple:
        """Generate flow instruction"""
        if instr.opcode == "select":
            dest = self.allocation.get(instr.result)
            cond = self._resolve_operand(instr.operands[0])
            true_val = self._resolve_operand(instr.operands[1])
            false_val = self._resolve_operand(instr.operands[2])
            return ("select", dest, cond, true_val, false_val)
        elif instr.opcode == "vselect":
            dest = self.allocation.get(instr.result)
            cond = self._resolve_operand(instr.operands[0])
            true_val = self._resolve_operand(instr.operands[1])
            false_val = self._resolve_operand(instr.operands[2])
            return ("vselect", dest, cond, true_val, false_val)
        else:
            raise ValueError(f"Unknown flow opcode: {instr.opcode}")


def generate(bundles: list[ScheduledBundle], allocation: Allocation) -> list[dict]:
    """Convenience function to generate code"""
    gen = CodeGenerator(bundles, allocation)
    return gen.generate()


class DirectCodeGenerator:
    """
    Alternative code generator that works directly with IR instructions,
    performing allocation on-the-fly. Useful for simpler cases.
    """

    def __init__(self, instructions: list[Instruction]):
        self.instructions = instructions
        self.allocation = Allocation()
        self.output: list[dict] = []

    def generate(self) -> tuple[list[dict], Allocation]:
        """Generate code with on-the-fly allocation"""
        # Pre-allocate all results
        for instr in self.instructions:
            if instr.result and instr.result not in self.allocation.value_to_addr:
                self.allocation.alloc(instr.result)

        # Generate one instruction per cycle (unscheduled baseline)
        for instr in self.instructions:
            slot = self._generate_slot(instr)
            if slot:
                engine_str = engine_to_str(instr.engine)
                self.output.append({engine_str: [slot]})

        return self.output, self.allocation

    def _resolve_operand(self, op) -> int:
        """Convert an operand to a scratch address"""
        if isinstance(op, Value):
            if op not in self.allocation.value_to_addr:
                self.allocation.alloc(op)
            return self.allocation.get(op)
        elif isinstance(op, Constant):
            return self.allocation.alloc_const(op.value)
        else:
            raise ValueError(f"Unknown operand type: {type(op)}")

    def _generate_slot(self, instr: Instruction) -> Optional[tuple]:
        """Generate a single instruction slot"""
        if instr.engine == Engine.ALU:
            dest = self.allocation.get(instr.result)
            src1 = self._resolve_operand(instr.operands[0])
            src2 = self._resolve_operand(instr.operands[1])
            return (instr.opcode, dest, src1, src2)

        elif instr.engine == Engine.VALU:
            if instr.opcode == "vbroadcast":
                dest = self.allocation.get(instr.result)
                src = self._resolve_operand(instr.operands[0])
                return ("vbroadcast", dest, src)
            else:
                dest = self.allocation.get(instr.result)
                src1 = self._resolve_operand(instr.operands[0])
                src2 = self._resolve_operand(instr.operands[1])
                return (instr.opcode, dest, src1, src2)

        elif instr.engine == Engine.LOAD:
            if instr.opcode == "const":
                dest = self.allocation.get(instr.result)
                value = instr.operands[0].value
                return ("const", dest, value)
            elif instr.opcode == "load":
                dest = self.allocation.get(instr.result)
                addr = self._resolve_operand(instr.operands[0])
                return ("load", dest, addr)
            elif instr.opcode == "vload":
                dest = self.allocation.get(instr.result)
                addr = self._resolve_operand(instr.operands[0])
                return ("vload", dest, addr)

        elif instr.engine == Engine.STORE:
            addr = self._resolve_operand(instr.operands[0])
            src = self._resolve_operand(instr.operands[1])
            if instr.opcode == "vstore":
                return ("vstore", addr, src)
            return ("store", addr, src)

        elif instr.engine == Engine.FLOW:
            if instr.opcode == "select":
                dest = self.allocation.get(instr.result)
                cond = self._resolve_operand(instr.operands[0])
                true_val = self._resolve_operand(instr.operands[1])
                false_val = self._resolve_operand(instr.operands[2])
                return ("select", dest, cond, true_val, false_val)
            elif instr.opcode == "vselect":
                dest = self.allocation.get(instr.result)
                cond = self._resolve_operand(instr.operands[0])
                true_val = self._resolve_operand(instr.operands[1])
                false_val = self._resolve_operand(instr.operands[2])
                return ("vselect", dest, cond, true_val, false_val)

        return None
