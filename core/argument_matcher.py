# core/argument_matcher.py
import os
import re
import logging
from typing import Any, Dict, Optional, Union, List
from datetime import datetime  # ✅ was missing — used in 'time' mode
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None
    logger.warning("torch not found. GPU support disabled.")


class ArgumentMatcher:
    """
    Matches tool call arguments based on schema-defined matching strategies.
    
    Supports per-field match modes: 'exact', 'normalized', 'semantic', 'time', 'fuzzy'.
    Handles partial predictions (missing/extra fields) gracefully for field-level metrics.
    """

    def __init__(
        self,
        tools_schema: List[Dict],  # Pass your tool definitions here
        global_match_mode: str = "semantic",  # fallback
        similarity_threshold: float = 0.85,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.global_match_mode = global_match_mode
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        self._has_embedding = False

        # Build schema map: {tool_name: {param_name: match_mode}}
        self.schema_map = {}
        for tool in tools_schema:
            tool_name = tool["function"]["name"]
            params = tool["function"]["parameters"]
            self.schema_map[tool_name] = {}
            for param_name, details in params.get("properties", {}).items():
                # Get match_mode from schema, or fallback to global
                match_mode = details.get("match_mode", global_match_mode)
                self.schema_map[tool_name][param_name] = match_mode

        # Initialize embeddings if needed
        if any("semantic" in modes.values() for modes in self.schema_map.values()) or global_match_mode == "semantic":
            try:
                from sentence_transformers import SentenceTransformer
                
                embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                logger.info(f"Using embedding model: {embedding_model_name}")
                
                if device is None:
                    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
                self.embedding_model = SentenceTransformer(
                    embedding_model_name,
                    device=device,
                    cache_folder=cache_dir
                )
                self._has_embedding = True
            except ImportError:
                logger.warning("sentence-transformers not found. Semantic matching falls back to normalized text.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{embedding_model_name}': {e}")
                self.embedding_model = None
                self._has_embedding = False

    def get_field_match_mode(self, tool_name: str, field_name: str) -> str:
        """Get match mode for a field, falling back to global mode."""
        return self.schema_map.get(tool_name, {}).get(field_name, self.global_match_mode)

    @staticmethod
    def normalize_text(s: str) -> str:
        if not isinstance(s, str):
            return str(s)
        s = s.lower()
        s = re.sub(r'[^\w\s\-\'/]', ' ', s)
        s = ' '.join(s.split())
        s = re.sub(r'(\d)\s*([ap]m)', r'\1 \2', s)
        return s

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _match_single_field(
        self,
        expected: Any,
        predicted: Any,
        tool_name: str,
        field_name: str,
        return_score: bool = False
    ) -> Union[bool, tuple[bool, Optional[float]]]:
        mode = self.get_field_match_mode(tool_name, field_name)

        if mode == "exact":
            result = expected == predicted
            return (result, None) if return_score else result

        elif mode == "normalized":
            if isinstance(expected, str) and isinstance(predicted, str):
                norm_eq = self.normalize_text(expected) == self.normalize_text(predicted)
                return (norm_eq, None) if return_score else norm_eq
            else:
                eq = expected == predicted
                return (eq, None) if return_score else eq

        elif mode == "semantic":
            if isinstance(expected, str) and isinstance(predicted, str) and self._has_embedding:
                try:
                    emb_exp, emb_pred = self.embedding_model.encode([expected, predicted])
                    sim = self.cosine_similarity(emb_exp, emb_pred)
                    match = sim >= self.similarity_threshold
                    return (match, sim) if return_score else match
                except Exception as e:
                    logger.warning(f"Embedding match failed for tool='{tool_name}', field='{field_name}': {e}")
                    # Fallback
                    norm_eq = self.normalize_text(expected) == self.normalize_text(predicted)
                    return (norm_eq, None) if return_score else norm_eq
            else:
                # Non-string or no embedding: fallback
                norm_eq = self.normalize_text(expected) == self.normalize_text(predicted)
                return (norm_eq, None) if return_score else norm_eq

        elif mode == "time":
            try:
                def parse_time(t: str) -> datetime:
                    t = t.strip().lower()
                    for fmt in ["%I%p", "%I:%M%p", "%H:%M", "%H"]:
                        try:
                            return datetime.strptime(t.replace(" ", ""), fmt)
                        except ValueError:
                            continue
                    raise ValueError(f"Could not parse: {t}")
                match = parse_time(str(expected)) == parse_time(str(predicted))
                return (match, None) if return_score else match
            except Exception as e:
                logger.warning(f"Time parsing failed for tool='{tool_name}', field='{field_name}': {e}")
                match = str(expected).lower() == str(predicted).lower()
                return (match, None) if return_score else match

        elif mode == "fuzzy":
            exp_norm = self.normalize_text(str(expected))
            pred_norm = self.normalize_text(str(predicted))
            if self._has_embedding:
                try:
                    emb_exp, emb_pred = self.embedding_model.encode([exp_norm, pred_norm])
                    sim = self.cosine_similarity(emb_exp, emb_pred)
                    match = sim >= self.similarity_threshold
                    return (match, sim) if return_score else match
                except Exception as e:
                    logger.warning(f"Fuzzy embedding match failed for tool='{tool_name}', field='{field_name}': {e}")
            # Fallback to substring
            match = exp_norm in pred_norm or pred_norm in exp_norm
            return (match, None) if return_score else match

        else:
            raise ValueError(f"Unknown match_mode '{mode}' for tool='{tool_name}', field='{field_name}'")

    def match_arguments(
        self,
        expected_args: Dict,
        predicted_args: Dict,
        tool_name: str,
        return_scores: bool = False
    ) -> Union[bool, Dict[str, Dict[str, Any]]]:
        """
        Match arguments for a tool call.
        
        If return_scores=False: returns bool (all expected fields correct?).
        If return_scores=True: returns detailed per-field results.
        Handles missing/extra fields gracefully.
        """
        expected_keys = set(expected_args.keys())
        predicted_keys = set(predicted_args.keys())
        common_keys = expected_keys & predicted_keys
        missing_in_pred = expected_keys - predicted_keys
        extra_in_pred = predicted_keys - expected_keys

        field_scores = {}
        overall_match = True

        # ✅ Evaluate common fields
        for key in common_keys:
            match, sim = self._match_single_field(
                expected_args[key], predicted_args[key], tool_name, key, return_score=True
            )
            mode = self.get_field_match_mode(tool_name, key)
            field_scores[key] = {
                "match": match,
                "similarity": sim,
                "mode": mode,
                "expected": expected_args[key],
                "predicted": predicted_args[key]
            }
            if not match:
                overall_match = False

        # ❌ Mark expected fields missing in prediction
        for key in missing_in_pred:
            mode = self.get_field_match_mode(tool_name, key)
            field_scores[key] = {
                "match": False,
                "similarity": None,
                "mode": mode,
                "expected": expected_args[key],
                "predicted": None,
                "reason": "missing in prediction"
            }
            overall_match = False

        # ⚠️ Mark extra fields (not expected)
        for key in extra_in_pred:
            field_scores[key] = {
                "match": False,
                "similarity": None,
                "mode": "unexpected",
                "expected": None,
                "predicted": predicted_args[key],
                "reason": "unexpected in prediction"
            }

        if return_scores:
            return {"overall_match": overall_match, "field_scores": field_scores}
        else:
            return overall_match

    def get_config(self) -> Dict[str, Any]:
        return {
            "global_match_mode": self.global_match_mode,
            "schema_based": bool(self.schema_map),
            "similarity_threshold": self.similarity_threshold,
            "has_embedding": self._has_embedding,
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        }