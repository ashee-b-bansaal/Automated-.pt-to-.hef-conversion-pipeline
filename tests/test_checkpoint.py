import unittest

from hailo_model_toolkit.checkpoint import clean_state_dict, extract_state_dict


class FakeTensor:
    shape = (1,)


class CheckpointTests(unittest.TestCase):
    def test_raw_state_dict(self):
        state = {"module.fc.weight": FakeTensor()}
        self.assertIs(extract_state_dict(state), state)
        cleaned = clean_state_dict(state)
        self.assertEqual(list(cleaned), ["fc.weight"])

    def test_nested_state_dict(self):
        state = {"fc.weight": FakeTensor()}
        self.assertIs(extract_state_dict({"model_state_dict": state}), state)


if __name__ == "__main__":
    unittest.main()
