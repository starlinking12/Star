"""
Binary vulnerability scanner using angr.
"""

import json
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict

try:
    import angr
    HAS_ANGR = True
except ImportError:
    HAS_ANGR = False


@dataclass
class BinaryVulnerability:
    address: int
    type: str
    severity: str
    description: str
    function: str
    instruction: str


class BinaryScanner:
    def __init__(self, binary_path: str):
        self.binary_path = Path(binary_path)
        self.project = None
    
    def load_binary(self):
        if not HAS_ANGR:
            print("angr not installed. Run: pip install angr")
            return False
        try:
            self.project = angr.Project(str(self.binary_path), auto_load_libs=False)
            return True
        except Exception as e:
            print(f"Failed to load: {e}")
            return False
    
    def scan(self) -> List[BinaryVulnerability]:
        if not self.load_binary():
            return []
        
        vulnerabilities = []
        dangerous_funcs = ["strcpy", "strcat", "sprintf", "gets", "scanf", "memcpy"]
        
        for func in self.project.kb.functions.values():
            for block in func.blocks:
                for insn in block.disassembly.insns:
                    for dangerous in dangerous_funcs:
                        if dangerous in insn.mnemonic or (insn.op_str and dangerous in insn.op_str):
                            vulnerabilities.append(BinaryVulnerability(
                                address=insn.address,
                                type="buffer_overflow",
                                severity="HIGH",
                                description=f"Unsafe function {dangerous}",
                                function=func.name,
                                instruction=insn.mnemonic + " " + (insn.op_str or "")
                            ))
                    break
        
        return vulnerabilities


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=str)
    parser.add_argument("--output", type=str, default="scan_results.json")
    args = parser.parse_args()
    
    scanner = BinaryScanner(args.binary)
    vulnerabilities = scanner.scan()
    
    results = {
        "binary": args.binary,
        "vulnerabilities": [asdict(v) for v in vulnerabilities],
        "total_found": len(vulnerabilities),
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Found {len(vulnerabilities)} vulnerabilities")


if __name__ == "__main__":
    main()