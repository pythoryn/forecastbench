"""Opt-in, read-only integration test for market-question category sampling.

Run this test explicitly with ``--question-set-date YYYY-MM-DD``. It downloads real
market-question data, applies the production filters and sampling workflow for that
creation date, and prints the resulting category distribution. Dataset sources and human
question sampling are outside this test's scope, and GCP uploads are forbidden.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from curate_questions.create_question_set import main as create_question_set
from helpers import constants, question_curation


def _parse_question_set_date(raw_date: str) -> date:
    """Parse an ISO date supplied on the pytest command line."""
    try:
        return date.fromisoformat(raw_date)
    except ValueError as exc:
        raise pytest.UsageError(
            f"--question-set-date must use YYYY-MM-DD; received {raw_date!r}."
        ) from exc


def _combine_market_questions(questions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine sampled market-source frames into one frame."""
    return pd.concat(questions.values(), ignore_index=True)


def _print_market_category_distribution(label: str, df: pd.DataFrame) -> None:
    """Print market-question counts and percentages for every allowed category."""
    allowed_categories = [
        category for category in constants.QUESTION_CATEGORIES if category != "Other"
    ]
    counts = df["category"].value_counts().reindex(allowed_categories, fill_value=0)
    distribution = pd.DataFrame(
        {
            "count": counts,
            "percent": (counts / len(df) * 100).round(1),
        }
    )

    print(f"\n{label} category distribution ({len(df)} questions)")
    print(distribution.to_string())
    print("\nCounts by source and category")
    print(pd.crosstab(df["source"], df["category"]).to_string())


def test_live_market_question_category_distribution(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create question sets from live data and analyze only market-question categories."""
    raw_date = request.config.getoption("--question-set-date")
    if raw_date is None:
        pytest.skip("Pass --question-set-date YYYY-MM-DD to run this live GCP test.")

    creation_date = _parse_question_set_date(raw_date)
    freeze_datetime = datetime.combine(creation_date, datetime.min.time(), tzinfo=timezone.utc)
    forecast_datetime = freeze_datetime + timedelta(days=question_curation.FREEZE_WINDOW_IN_DAYS)

    if not create_question_set.env.QUESTION_BANK_BUCKET:
        pytest.fail("QUESTION_BANK_BUCKET is unset. Load variables.mk before running this test.")

    monkeypatch.setattr(question_curation, "FREEZE_DATETIME", freeze_datetime)
    monkeypatch.setattr(question_curation, "FORECAST_DATETIME", forecast_datetime)
    monkeypatch.setattr(question_curation, "FORECAST_DATE", forecast_datetime.date())

    # The production helper uses fixed /tmp filenames and can therefore read an old local
    # file after a failed download. Redirect every download to this test's new temporary
    # directory so a passing run is guaranteed to have read the bucket during this run.
    def forbid_upload(*args: object, **kwargs: object) -> None:
        pytest.fail("The read-only live test attempted to upload data to GCP.")

    monkeypatch.setattr(create_question_set.gcp.storage, "upload", forbid_upload)

    print(f"\nQuestion bank bucket: {create_question_set.env.QUESTION_BANK_BUCKET}")
    print(f"Simulated question-set creation date: {creation_date.isoformat()}")
    print(f"Resulting forecast due date: {forecast_datetime.date().isoformat()}")

    original_download_and_read = create_question_set.data_utils.download_and_read
    with TemporaryDirectory(prefix=".question-set-live-", dir=Path.cwd()) as temp_dir:

        def download_and_read_fresh(
            filename: str, local_filename: str, df_tmp: pd.DataFrame, dtype: dict
        ) -> pd.DataFrame:
            fresh_local_filename = Path(temp_dir) / Path(local_filename).name
            return original_download_and_read(
                filename=filename,
                local_filename=str(fresh_local_filename),
                df_tmp=df_tmp,
                dtype=dtype,
            )

        monkeypatch.setattr(
            create_question_set.data_utils, "download_and_read", download_and_read_fresh
        )

        dfmeta = create_question_set.data_utils.download_and_read(
            filename=constants.META_DATA_FILENAME,
            local_filename=f"/tmp/{constants.META_DATA_FILENAME}",
            df_tmp=pd.DataFrame(columns=constants.META_DATA_FILE_COLUMNS).astype(
                constants.META_DATA_FILE_COLUMN_DTYPE
            ),
            dtype=constants.META_DATA_FILE_COLUMN_DTYPE,
        )

        market_questions = {}
        for source, source_config in question_curation.FREEZE_QUESTION_MARKET_SOURCES.items():
            dfq = create_question_set.data_utils.get_data_from_cloud_storage(
                source=source,
                return_question_data=True,
            )
            if dfq.empty:
                pytest.fail(f"Found no live market questions for {source}.")

            dfq["source"] = source
            dfq = create_question_set.drop_invalid_questions(dfq=dfq, dfmeta=dfmeta)
            dfq = create_question_set.drop_missing_freeze_datetime(dfq)
            dfq = dfq[dfq["category"] != "Other"]
            dfq = dfq[~dfq["resolved"]]
            dfq = create_question_set.drop_questions_that_resolve_too_soon(source=source, dfq=dfq)
            market_questions[source] = {
                **source_config,
                "dfq": dfq.reset_index(drop=True),
                "num_questions_available": len(dfq),
            }

        allocations = create_question_set.allocate_across_sources(
            questions=market_questions,
            num_questions=question_curation.FREEZE_NUM_LLM_QUESTIONS // 2,
        )
        sampled_market_questions = create_question_set.sample_market_questions_across_sources(
            questions=market_questions,
            allocations=allocations,
        )

    market_df = _combine_market_questions(sampled_market_questions)
    allowed_categories = set(constants.QUESTION_CATEGORIES) - {"Other"}
    unexpected_categories = set(market_df["category"]) - allowed_categories

    assert len(market_df) == question_curation.FREEZE_NUM_LLM_QUESTIONS // 2
    assert set(market_df["source"]) == set(question_curation.MARKET_SOURCES)
    assert not unexpected_categories, f"Unexpected categories: {sorted(unexpected_categories)}"
    _print_market_category_distribution(
        f"{creation_date.isoformat()} LLM MARKET QUESTIONS", market_df
    )
