"""Check that bevformer's PCCREvalMixin is a verbatim copy of PCCR's BEVDet adapter members.

    python3 tools/check_pccr_eval_copy.py

Compares the AST of every class attribute / method of PCCREvalMixin
(bevformer/projects/mmdet3d_plugin/datasets/pccr_eval.py) with the member of the same name in
class PCCRDataset of third_party/pccr/models/BEVDet/pccr_dataset.py (the PCCR benchmark's
adapter, also copied verbatim into bevdepth/). Line numbers and comments are ignored.
Exit code 1 on any difference.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'third_party/pccr/models/BEVDet/pccr_dataset.py'
COPIES = {
    REPO / 'bevformer/projects/mmdet3d_plugin/datasets/pccr_eval.py': 'PCCREvalMixin',
}


def members(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    out = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node
    return out


def main():
    source = members(SOURCE, 'PCCRDataset')
    failed = False
    for path, class_name in COPIES.items():
        copy = members(path, class_name)
        for name, node in copy.items():
            if name not in source:
                print(f'FAIL {path.name}:{class_name}.{name} does not exist in the PCCR adapter')
                failed = True
                continue
            if ast.dump(node) != ast.dump(source[name]):
                print(f'FAIL {path.name}:{class_name}.{name} differs from the PCCR adapter')
                failed = True
            else:
                print(f'ok   {class_name}.{name}')
    print('all checks passed' if not failed else 'differences found')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
