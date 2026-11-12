"""CPU checks against the actual notebook; no Colab, tracker, or network needed."""
import ast
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import cv2
import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/rock_pick_autolabel.ipynb'


def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def helpers() -> dict:
    """Load constants and pure helpers without importing torch or starting Colab."""
    tree = ast.parse(''.join(notebook()['cells'][2]['source']))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    namespace = {'np': np, 'cv2': cv2, 'os': os, 'Path': Path, 'math': math}
    exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
    return namespace


class NotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = helpers()

    def make_pair(self, root: Path, clip: str, frame: int, visibility: int = 2) -> None:
        for sub in ('images', 'labels'):
            (root / 'train' / sub).mkdir(parents=True, exist_ok=True)
        name = f'{clip}_f{frame:05d}'
        self.assertTrue(cv2.imwrite(str(root / 'train/images' / f'{name}.jpg'),
                                   np.zeros((16, 16, 3), dtype=np.uint8)))
        parts = ['0', '0.5', '0.5', '0.5', '0.5'] + ['0.5', '0.5', str(visibility)] * 5
        (root / 'train/labels' / f'{name}.txt').write_text(' '.join(parts) + '\n')

    def test_clean_metadata_and_code_syntax(self) -> None:
        doc = notebook()
        self.assertEqual((doc['nbformat'], doc['nbformat_minor']), (4, 5))
        self.assertEqual(len(doc['cells']), 8)
        self.assertNotIn('widgets', doc['metadata'])
        for cell in doc['cells']:
            self.assertEqual(cell['metadata'], {})
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                # Colab's %pip line is IPython syntax; all remaining code is Python.
                code = ''.join(line for line in cell['source'] if not line.startswith('%pip'))
                compile(code, str(NOTEBOOK), 'exec')

    def test_crop_padding_and_coordinate_translation(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        points = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=float)
        image, translated, visible, contains_all = self.ns['crop_around_pick'](
            frame, points, np.ones(5, dtype=bool))
        self.assertEqual(image.shape, (480, 640, 3))
        np.testing.assert_array_equal(translated, points + [170, 140])
        self.assertTrue(visible.all())
        self.assertTrue(contains_all)

    def test_crop_marks_keypoints_outside_saved_window_invisible(self) -> None:
        frame = np.zeros((800, 1200, 3), dtype=np.uint8)
        points = np.array([[0, 200], [1199, 200], [500, 300], [600, 400], [700, 500]], dtype=float)
        image, translated, visible, contains_all = self.ns['crop_around_pick'](
            frame, points, np.ones(5, dtype=bool))
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertFalse(contains_all)
        self.assertFalse(visible[0])
        self.assertFalse(visible[1])
        self.assertTrue(visible[2:].all())
        np.testing.assert_array_equal(translated[2], [220, 190])

    def test_yolo_fields_and_invalid_points(self) -> None:
        points = np.array([[10, 10], [100, 100], [np.nan, 0], [-2, 30], [640, 40]])
        fields = self.ns['yolo_line'](points, np.ones(5, dtype=bool), 640, 480, .08).split()
        self.assertEqual(len(fields), 20)
        values = np.array(fields, dtype=float)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values[1:5] >= 0) & (values[1:5] <= 1)).all())
        np.testing.assert_array_equal(values[7::3], [2, 2, 0, 0, 0])
        for offset in (11, 14, 17):
            np.testing.assert_array_equal(values[offset:offset+3], [0, 0, 0])
        with self.assertRaises(ValueError):
            self.ns['yolo_line'](points, np.zeros(5, dtype=bool), 640, 480, .08)
        with self.assertRaises(ValueError):
            self.ns['yolo_line'](np.full((5, 2), 30), np.ones(5, dtype=bool), 640, 480, .08)

    def test_black_border_estimation_on_synthetic_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'border.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (160, 120))
            if not writer.isOpened():
                self.skipTest('OpenCV MJPG writer is unavailable')
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[20:100, 24:136] = 200
            for _ in range(4):
                writer.write(frame)
            writer.release()
            x0, y0, x1, y1 = self.ns['detect_content_box'](path)
            self.assertLessEqual(x0, 24)
            self.assertLessEqual(y0, 20)
            self.assertGreaterEqual(x1, 136)
            self.assertGreaterEqual(y1, 100)
            self.assertGreater(x0, 0)
            self.assertLess(x1, 160)

    def test_reject_pairs_and_export_only_current_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'out'
            export = Path(temp) / 'Annotated'
            export.mkdir()
            self.make_pair(root, 'red-1', 0)
            self.make_pair(root, 'red-1', 8, visibility=0)
            self.make_pair(root, 'red-2', 0)
            self.ns.update({'OUT': str(root), 'CLIP': 'red-1', 'shutil': shutil,
                            'ANNOTATED_DIR': export})
            self.ns['write_data_yaml']()
            config = (root / 'data.yaml').read_text()
            self.assertIn('val: val/images', config)
            self.assertNotIn('val: train/images', config)
            self.assertNotIn('path: .', config)
            exec(compile(''.join(notebook()['cells'][6]['source']), 'review', 'exec'), self.ns)
            self.assertFalse((root / 'train/images/red-1_f00008.jpg').exists())
            self.assertTrue((root / 'rejected/images/red-1_f00008.jpg').exists())
            self.assertTrue((root / 'rejected/labels/red-1_f00008.txt').exists())
            self.assertTrue((root / 'train/images/red-2_f00000.jpg').exists())
            code = compile(''.join(notebook()['cells'][7]['source']), 'export', 'exec')
            exec(code, self.ns)
            with zipfile.ZipFile(self.ns['ZIP_PATH']) as archive:
                files = {name for name in archive.namelist() if not name.endswith('/')}
                self.assertEqual(files, {'train/images/red-1_f00000.jpg',
                                         'train/labels/red-1_f00000.txt', 'data.yaml'})
            with self.assertRaises(AssertionError):
                exec(code, self.ns)  # Existing ZIP must not be overwritten.

    def test_malformed_review_moves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_pair(root, 'red-1', 0, visibility=0)
            self.make_pair(root, 'red-1', 8)
            (root / 'train/labels/red-1_f00008.txt').write_text('malformed')
            self.ns.update({'OUT': str(root), 'CLIP': 'red-1', 'shutil': shutil})
            with self.assertRaises(ValueError):
                exec(compile(''.join(notebook()['cells'][6]['source']), 'review', 'exec'), self.ns)
            self.assertTrue((root / 'train/images/red-1_f00000.jpg').exists())
            self.assertFalse((root / 'rejected').exists())


if __name__ == '__main__':
    unittest.main()
