import random
from typing import List


def sample_qids(all_qids: List[str], max_queries: int, seed: int) -> List[str]:
    if max_queries <= 0 or len(all_qids) <= max_queries:
        return all_qids
    rng = random.Random(seed)
    return rng.sample(all_qids, max_queries)

