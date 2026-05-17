import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.similarity_search_service import find_similarity_hits  # noqa: E402


class SimilaritySearchServiceTests(unittest.TestCase):
    def test_similarity_search_returns_ranked_hits(self):
        hits = find_similarity_hits(["CCO"], ["CCO", "CCN", "c1ccccc1"], top_k=2, include_self=True)
        self.assertGreaterEqual(len(hits), 1)
        self.assertGreaterEqual(hits[0].similarity, hits[-1].similarity)


if __name__ == "__main__":
    unittest.main()
