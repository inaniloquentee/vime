class _ReplayManager:
    enabled = False
    stage = 'fallthrough'
    enable_check_replay_result = False

    def get_replay_indices(self, n_tokens, topk):
        return None

    def record_indices(self, indices):
        return None

    def get_topk_fn(self, old_topk_fn, return_probs):
        """Return the original router top-k function when replay is disabled."""
        if self.enabled:
            raise RuntimeError("VIME replay manager does not implement enabled routing replay")
        return old_topk_fn

    def register_to_module(self, *args, **kwargs):
        return None

    def check_replay_indices(self, *args, **kwargs):
        return None

    @staticmethod
    def fill_replay_padding(indices, num_scores):
        all_invalid = (indices == -1).all(dim=-1)
        ar = __import__('torch').arange(indices.shape[-1], device=indices.device, dtype=indices.dtype) % num_scores
        return __import__('torch').where(all_invalid.unsqueeze(-1), ar, indices)

indexer_replay_manager = _ReplayManager()

routing_replay_manager = indexer_replay_manager
