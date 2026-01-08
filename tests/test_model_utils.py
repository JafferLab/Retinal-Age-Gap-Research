import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add webapp to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../webapp'))

from model_utils import RetinalAgeModel, RECALIB_SLOPE, RECALIB_INTERCEPT

class TestRetinalAgeModel(unittest.TestCase):
    def setUp(self):
        self.model = RetinalAgeModel(model_path="dummy_path")
        
        # Mock the session
        self.mock_session = MagicMock()
        self.model.session = self.mock_session
        self.model.input_name = "input"
        
        # Determine expected values based on CONSTANTS in model_utils
        self.slope = RECALIB_SLOPE
        self.intercept = RECALIB_INTERCEPT

        # Patch os.path.exists and os.path.getsize to avoid real file checks
        self.patcher_exists = patch('model_utils.os.path.exists', return_value=True)
        self.patcher_getsize = patch('model_utils.os.path.getsize', return_value=120*1024*1024) # 120MB
        
        self.mock_exists = self.patcher_exists.start()
        self.mock_getsize = self.patcher_getsize.start()

    def tearDown(self):
        self.patcher_exists.stop()
        self.patcher_getsize.stop()

    @patch('model_utils.onnxruntime.InferenceSession')
    def test_predict_original_mode(self, mock_session_cls):
        """Test that 'original' mode returns the raw model output."""
        # Setup
        raw_output = 45.0
        self.mock_session.run.return_value = [np.array([raw_output])]
        
        # Create a dummy image
        from PIL import Image
        dummy_img = Image.new('RGB', (100, 100))
        
        # Execute
        # We expect the model (assuming it's JOIR) to output 45.0
        # In 'original' mode, we should get 45.0 back
        prediction = self.model.predict(dummy_img, laterality='OD', recalibration_mode='original')
        
        # Verify
        self.assertEqual(prediction, 45.0)

    @patch('model_utils.onnxruntime.InferenceSession')
    def test_predict_chinese_mode(self, mock_session_cls):
        """Test that 'chinese' mode applies the linear transformation."""
        # Setup
        raw_output = 45.0
        self.mock_session.run.return_value = [np.array([raw_output])]
        
        # Create a dummy image
        from PIL import Image
        dummy_img = Image.new('RGB', (100, 100))
        
        # Execute
        # In 'chinese' mode, we expect (Raw * Slope) + Intercept
        expected_age = (raw_output * self.slope) + self.intercept
        # Rounding logic in app is round(x, 1)
        expected_age = round(expected_age, 1)
        
        prediction = self.model.predict(dummy_img, laterality='OD', recalibration_mode='chinese')
        
        # Verify
        self.assertEqual(prediction, expected_age)

if __name__ == '__main__':
    unittest.main()
