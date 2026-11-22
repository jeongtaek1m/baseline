"""Check the vanilla-BEVDepth code path in bevdepth/ against upstream BEVDet dev2.0 @2ae6a02.

    git clone https://github.com/HuangJunJie2017/BEVDet /tmp/BEVDet && git -C /tmp/BEVDet checkout 2ae6a02
    python tools/check_bevdepth_upstream.py /tmp/BEVDet

1. The restored copies (`*_upstream.py`) must be byte-identical to upstream except for the
   `register_module(name='<Name>Upstream')` line.
2. For every other class the vanilla config builds from PD-BEV's own files, PD-BEV's source
   must equal upstream up to whitespace and comments. Known, reviewed exceptions are listed
   in ALLOWED_DIFFS with the reason.
"""
import ast
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEVDEPTH = os.path.join(HERE, '..', 'bevdepth')

RESTORED = {
    'mmdet3d/datasets/pipelines/loading_upstream.py': (
        'mmdet3d/datasets/pipelines/loading.py',
        ['mmlabNormalize', 'PointToMultiViewDepth', 'PrepareImageInputs', 'LoadAnnotationsBEVDepth']),
    'mmdet3d/models/dense_heads/centerpoint_head_upstream.py': (
        'mmdet3d/models/dense_heads/centerpoint_head.py', ['CenterHead']),
}

# Classes the vanilla config takes from PD-BEV's files as they are.
SHARED = {
    'mmdet3d/models/detectors/bevdet.py': ['BEVDet', 'BEVDet4D', 'BEVDepth4D'],
    'mmdet3d/models/necks/view_transformer.py': ['LSSViewTransformer', 'LSSViewTransformerBEVDepth', 'DepthNet',
                                                 'Mlp', 'SELayer', 'ASPP', '_ASPPModule'],
    'mmdet3d/models/necks/fpn.py': ['CustomFPN'],
    'mmdet3d/models/necks/lss_fpn.py': ['FPN_LSS'],
    'mmdet3d/models/backbones/resnet.py': ['CustomResNet'],
    'mmdet3d/models/dense_heads/centerpoint_head.py': ['SeparateHead', 'DCNSeparateHead'],
    'mmdet3d/core/hook/ema.py': ['MEGVIIEMAHook'],
    'mmdet3d/datasets/pipelines/loading.py': ['LoadPointsFromFile'],
}

# PD-BEV changes that remain on the vanilla path, reviewed by hand (see docs/porting_plan.md).
ALLOWED_DIFFS = {
    'BEVDet': 'simple_test also returns img_metas; PCCRDataset.evaluate (via NuScenesDataset.form_meta) '
              'drops them before formatting. Boxes are unchanged.',
    'DepthNet': 'adds mlp_channels=27 (same default) and a squeeze(-1) that is a no-op for the (B, N, 27) '
                'mlp_input built by LSSViewTransformerBEVDepth.get_mlp_input.',
}


def defs(path):
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            out[node.name] = ''.join(lines[start - 1:node.end_lineno])
    return out


def strip_ws_comments(seg):
    out = []
    for line in seg.splitlines():
        code = line.split('#', 1)[0] if not line.lstrip().startswith(('"', "'")) else line
        code = ' '.join(code.split())
        if code:
            out.append(code)
    return out


def main(upstream):
    ok = True
    print('== restored copies (must be byte-identical except the registry name)')
    for ours, (theirs, names) in RESTORED.items():
        a, b = defs(os.path.join(upstream, theirs)), defs(os.path.join(BEVDEPTH, ours))
        for n in names:
            up = a[n]
            mine = b.get(n)
            if mine is not None and up.startswith('@'):
                up_dec, up_body = up.split('\n', 1)
                my_dec, my_body = mine.split('\n', 1)
                same = up_body == my_body and my_dec == up_dec.replace('()', f"(name='{n}Upstream')")
            else:
                same = mine == up
            ok &= same
            print(f'   {"OK  " if same else "FAIL"} {n:26s} {ours}')

    print('== classes shared with PD-BEV (must equal upstream up to whitespace/comments)')
    for path, names in SHARED.items():
        a, b = defs(os.path.join(upstream, path)), defs(os.path.join(BEVDEPTH, path))
        for n in names:
            same = strip_ws_comments(a[n]) == strip_ws_comments(b[n])
            if same:
                print(f'   OK   {n:26s} {path}')
            elif n in ALLOWED_DIFFS:
                print(f'   KNOWN {n:25s} {path}: {ALLOWED_DIFFS[n]}')
            else:
                ok = False
                print(f'   FAIL {n:26s} {path}')
                for line in difflib.unified_diff(strip_ws_comments(a[n]), strip_ws_comments(b[n]), lineterm='', n=0):
                    print('        ' + line)
    print('RESULT:', 'all checks passed' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
