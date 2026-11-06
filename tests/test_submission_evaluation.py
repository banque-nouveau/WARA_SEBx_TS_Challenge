import os
import unittest

import numpy as np

from amgm.data import load_test_data
from amgm.data.load_test_data import PredictionInputs
from amgm.models import mymodel


class SubmissionInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = load_test_data(
            os.environ.get("SUBMISSION_TEST_DATA", "Data/data-20250901"),
            start_date=os.environ.get("SUBMISSION_TEST_START_DATE", "2018-12-31"),
            end_date=os.environ.get("SUBMISSION_TEST_END_DATE", "2020-12-31"),
            num_iids=int(os.environ.get("SUBMISSION_TEST_NUM_IIDS", "500")),
            seed=int(os.environ.get("SUBMISSION_TEST_SEED", "1")),
        )

    def test_model_constructor_and_prediction_format(self):
        self.assertTrue(callable(getattr(mymodel, "model", None)), "mymodel.model must be callable")
        model = mymodel.model()
        self.assertTrue(callable(getattr(model, "predict", None)), "model.predict must be callable")

        inputs = self.test_data.inputs
        self.assertIsInstance(inputs, PredictionInputs)
        predictions = np.asarray(model.predict(inputs))
        n = len(inputs.features)
        self.assertIn(predictions.shape, ((n,), (n, 1)), "Return one prediction per input sample")
        self.assertTrue(
            np.issubdtype(predictions.dtype, np.integer) or np.issubdtype(predictions.dtype, np.floating),
            "Predictions must be real numbers",
        )
        self.assertTrue(np.isfinite(predictions).all(), "Predictions must be finite")


if __name__ == "__main__":
    unittest.main()
