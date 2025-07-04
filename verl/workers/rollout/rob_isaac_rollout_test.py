import unittest

from multiprocessing import Queue

from rob_isaac_rollout import env_worker


class TestEnvWorker(unittest.TestCase):
    def test_env_worker(self):
        input_q = Queue()
        output_q = Queue()
        is_valid = True
        global_step = 0
        max_steps = 512
        env_worker("PnPCounterToCab", 0, 0, {}, input_q, output_q, is_valid, global_step, max_steps)


if __name__ == "__main__":
    test = TestEnvWorker()
    test.test_env_worker()
