from .hotpot_eval import remap_supporting_facts_to_titles, score_hotpot_predictions
from .squad_eval import score_squad_predictions

__all__ = ["score_hotpot_predictions", "remap_supporting_facts_to_titles", "score_squad_predictions"]
