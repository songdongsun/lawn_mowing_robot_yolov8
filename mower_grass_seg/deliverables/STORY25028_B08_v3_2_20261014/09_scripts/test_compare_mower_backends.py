import unittest

import numpy as np

from compare_mower_backends import decode, measure, prepare


class DecoderTests(unittest.TestCase):
    def test_empty_masks_are_explicit(self):
        mask = np.zeros((2, 2), dtype=bool)
        self.assertEqual(measure(mask, mask)['mask_iou'], 1)
        self.assertTrue(measure(mask, mask)['both_empty'])
        self.assertEqual(measure(mask, ~mask)['mask_iou'], 0)

    def test_letterbox(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, geom = prepare(image)
        self.assertEqual(geom, (480, 640, 1, 640, 480, 0, 80))
        self.assertTrue((padded[:80] == 114).all())
        self.assertTrue((padded[80:560] == 0).all())

    def test_confidence_boundary_nms_and_box_crop(self):
        pred = np.zeros((37, 8400), dtype=np.float32)
        proto = np.zeros((32, 160, 160), dtype=np.float32)
        proto[0] = 1
        for anchor, score in enumerate([.25, .3, .249]):
            pred[:4, anchor] = [100, 180, 20, 20]
            pred[4, anchor] = score
            pred[5, anchor] = 1
        mask, details = decode(pred, proto, (480, 640, 1, 640, 480, 0, 80))
        self.assertEqual(details['candidates'], 2)
        self.assertEqual(len(details['detections']), 1)
        self.assertEqual(int(mask.sum()), 400)
        self.assertTrue(mask[90:110, 90:110].all())

    def test_invalid_shape_rejected(self):
        with self.assertRaises(AssertionError):
            decode(np.zeros((36, 8400)), np.zeros((32, 160, 160)), ())

    def test_outward_crop_keeps_image_edges(self):
        pred = np.zeros((37, 8400), dtype=np.float32)
        proto = np.ones((32, 160, 160), dtype=np.float32)
        pred[:4, 0] = [320, 320, 650, 650]
        pred[4, 0] = .9
        pred[5, 0] = 1
        mask, _ = decode(pred, proto, (480, 640, 1, 640, 480, 0, 80), True)
        self.assertTrue(mask.all())

    def test_outward_crop_fractional_endpoints(self):
        pred = np.zeros((37, 8400), dtype=np.float32)
        proto = np.ones((32, 160, 160), dtype=np.float32)
        pred[:4, 0] = [321.5, 420.5, 640.5, 270.5]
        pred[4, 0] = .9
        pred[5, 0] = 1
        mask, _ = decode(pred, proto, (480, 640, 1, 640, 480, 0, 80), True)
        self.assertTrue(mask[475, 1:].all())
        self.assertFalse(mask[476:].any())


if __name__ == '__main__':
    unittest.main()
