import unittest

from multiprocessing import Queue
from omegaconf import OmegaConf

from rob_rollout import env_worker


class TestEnvWorker(unittest.TestCase):
    def test_env_worker(self):
        task_name = "libero_10"
        t_id = 8
        tr_id = 14
        config = OmegaConf.create({
            'vla': 'openvla-oft', 
            'action_chunks_len': 8, 
            'experiment_name': 'debug', 
            'unnorm_key': 'libero_10_no_noops', 
            'micro_batch_size': 1, 
            'val_micro_batch_size': 8, 
            'model_family': 'openvla', 
            'task_suite_name': 'libero_10', 
            'num_steps_wait': 10, 
            'pretrained_checkpoint': 'Haozhan72/Openvla-oft-SFT-libero10-trajall', 
            'center_crop': True, 
            'max_prompt_length': 512, 
            'num_images_in_input': 1, 
            'name': 'hf', 
            'temperature': 1.6, 
            'top_k': -1, 
            'top_p': 1, 
            'prompt_length': 256, 
            'response_length': 128, 
            'dtype': 'bfloat16', 
            'gpu_memory_utilization': 0.9, 
            'ignore_eos': False, 
            'enforce_eager': True, 
            'free_cache_engine': True, 
            'load_format': 'dummy_dtensor', 
            'tensor_model_parallel_size': 1, 
            'max_num_batched_tokens': 8192, 
            'max_num_seqs': 1024, 
            'log_prob_micro_batch_size': 4, 
            'log_prob_use_dynamic_bsz': False, 
            'log_prob_max_token_len_per_gpu': 20480, 
            'do_sample': True, 
            'n': 1
        })
        input_q = Queue()
        output_q = Queue()
        is_valid = True
        global_step = 0
        max_steps = 512

        env_worker(task_name, t_id, tr_id, config, input_q, output_q, is_valid, global_step, max_steps)


if __name__ == "__main__":
    test = TestEnvWorker()
    test.test_env_worker()
