"""Tests for create_question_set market sampling.

Focus: the soft category balancing added to `stratified_sample_questions`. The contract is
that market questions get spread across categories wherever the data allows, without ever
disturbing the market-value/time-horizon (composite-bin) distribution that stratified
sampling exists to enforce.
"""

from datetime import timedelta

import pandas as pd

from curate_questions.create_question_set.main import (
    sample_market_questions_across_sources,
    stratified_sample_questions,
)
from helpers import question_curation

CATS4 = [
    "Politics & Governance",
    "Sports",
    "Economics & Business",
    "Science & Tech",
]


def _make_binned_df(bin_specs):
    """Build a dfq for stratified_sample_questions.

    Args:
        bin_specs (list): (composite_bin, bin_weight, {category: count}) tuples

    Returns
        dfq (pd.DataFrame): Rows with id, composite_bin, bin_weight and category columns
    """
    rows = []
    qid = 0
    for composite_bin, bin_weight, category_counts in bin_specs:
        for category, count in category_counts.items():
            for _ in range(count):
                rows.append(
                    {
                        "id": f"q{qid}",
                        "composite_bin": composite_bin,
                        "bin_weight": bin_weight,
                        "category": category,
                    }
                )
                qid += 1
    return pd.DataFrame(rows)


def _make_market_df(source, category_counts):
    """Build market questions that all belong to the same composite bin."""
    close_datetime = (question_curation.FORECAST_DATETIME + timedelta(days=20)).isoformat()
    rows = []
    for category, count in category_counts.items():
        for index in range(count):
            rows.append(
                {
                    "id": f"{source}-{category}-{index}",
                    "source": source,
                    "category": category,
                    "freeze_datetime_value": 0.5,
                    "market_info_close_datetime": close_datetime,
                }
            )
    return pd.DataFrame(rows)


class TestCategoryBalancedStratifiedSampling:
    """Category balancing must not come at the expense of the composite-bin targets."""

    def test_balances_categories_and_preserves_bin_distribution(self):
        """Balance abundant categories without changing composite-bin counts."""
        # Two equally weighted market-value/time-horizon bins, each holding a surplus of
        # every category. The sampler must keep the 50/50 split across bins while spreading
        # the selection evenly over the four categories.
        df = _make_binned_df(
            [
                ("mvA_thA", 0.5, {c: 5 for c in CATS4}),
                ("mvB_thB", 0.5, {c: 5 for c in CATS4}),
            ]
        )
        result = stratified_sample_questions(df, n_target=8)

        assert len(result) == 8
        assert result["composite_bin"].value_counts().to_dict() == {"mvA_thA": 4, "mvB_thB": 4}
        # 8 questions across 4 abundant categories -> exactly 2 each.
        assert result["category"].value_counts().to_dict() == {c: 2 for c in CATS4}

    def test_category_counts_within_one_when_not_divisible(self):
        """Keep category counts within one when an exact split is impossible."""
        # 8 questions across 3 abundant categories cannot be exactly even; the spread must
        # still be as flat as possible (no category more than one ahead of another).
        df = _make_binned_df(
            [
                (
                    "mvA_thA",
                    1.0,
                    {"Sports": 10, "Politics & Governance": 10, "Economics & Business": 10},
                )
            ]
        )
        result = stratified_sample_questions(df, n_target=8)

        counts = result["category"].value_counts()
        assert len(result) == 8
        assert counts.max() - counts.min() <= 1

    def test_bin_distribution_preserved_when_categories_are_forced(self):
        """Preserve composite-bin counts when category choices are constrained."""
        # Each bin offers only a single, different category, so an even category spread is
        # impossible. The sampler must still honor the composite-bin counts rather than
        # distort the market-value/time-horizon distribution chasing category balance.
        df = _make_binned_df(
            [
                ("mvA_thA", 0.5, {"Sports": 10}),
                ("mvB_thB", 0.5, {"Politics & Governance": 10}),
            ]
        )
        result = stratified_sample_questions(df, n_target=8)

        assert result["composite_bin"].value_counts().to_dict() == {"mvA_thA": 4, "mvB_thB": 4}

    def test_joint_allocation_offsets_a_constrained_bin(self):
        """Use flexible-bin choices to offset a constrained bin."""
        # The flexible slots should offset the other bin's unavoidable category
        # concentration, regardless of which bin appears first in the input.
        df = _make_binned_df(
            [
                ("flexible", 0.5, {"Sports": 5, "Politics & Governance": 5}),
                ("forced", 0.5, {"Sports": 5}),
            ]
        )
        result = stratified_sample_questions(df, n_target=4)

        assert result["composite_bin"].value_counts().to_dict() == {
            "flexible": 2,
            "forced": 2,
        }
        assert result["category"].value_counts().to_dict() == {
            "Sports": 2,
            "Politics & Governance": 2,
        }

    def test_joint_allocation_handles_overlapping_category_choices(self):
        """Find the global balance when category choices overlap across bins."""
        # Looking only at current counts can choose Sports from the middle bin and leave
        # no way for the last bin to add Politics. Joint allocation sees all three choices.
        df = _make_binned_df(
            [
                ("forced_science", 1 / 3, {"Science & Tech": 2}),
                ("politics_or_sports", 1 / 3, {"Politics & Governance": 2, "Sports": 1}),
                ("sports_or_science", 1 / 3, {"Sports": 2, "Science & Tech": 2}),
            ]
        )
        result = stratified_sample_questions(df, n_target=3)

        assert result["composite_bin"].value_counts().to_dict() == {
            "forced_science": 1,
            "politics_or_sports": 1,
            "sports_or_science": 1,
        }
        assert result["category"].value_counts().to_dict() == {
            "Science & Tech": 1,
            "Politics & Governance": 1,
            "Sports": 1,
        }

    def test_balances_categories_across_market_sources(self):
        """Balance categories jointly while preserving each market-source quota."""
        # The flexible source's selections should compensate for the category forced by the
        # other source, while both source quotas stay fixed.
        questions = {
            "metaculus": {
                "dfq": _make_market_df("metaculus", {"Sports": 4, "Politics & Governance": 4})
            },
            "manifold": {"dfq": _make_market_df("manifold", {"Sports": 4})},
        }
        allocations = {
            "metaculus": {"num_questions_to_sample": 2},
            "manifold": {"num_questions_to_sample": 2},
        }

        result = sample_market_questions_across_sources(questions, allocations)
        combined = pd.concat(result.values(), ignore_index=True)

        assert {source: len(df) for source, df in result.items()} == {
            "metaculus": 2,
            "manifold": 2,
        }
        assert combined["category"].value_counts().to_dict() == {
            "Sports": 2,
            "Politics & Governance": 2,
        }
