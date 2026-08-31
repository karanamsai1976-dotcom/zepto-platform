"""Console entry point for model card generation: zepto-card.

The qualitative sections live here as text rather than being generated, because
judgements about intended use and ethical risk are claims a person makes and
should be attributable to one. The quantitative sections are computed from the
stored artifact so they cannot drift from the model they describe.
"""

from __future__ import annotations

from pathlib import Path

from zepto.analytics.datasets import load_titanic
from zepto.analytics.features import build_features
from zepto.analytics.model_card import compare_to_baseline, evaluate_by_group, generate_card
from zepto.analytics.registry import ModelRegistry
from zepto.analytics.settings import AnalyticsSettings, get_analytics_settings
from zepto.analytics.training import split_data
from zepto.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

CARD_FILENAME = "MODEL_CARD.md"

#: Columns the disaggregated analysis slices on. Chosen because they are the
#: model's dominant predictors and the attributes whose disparate treatment
#: would matter most if this were ever deployed against people.
GROUP_COLUMNS = ("sex", "pclass")

DATASET_DESCRIPTION = """\
The seaborn distribution of the Titanic passenger manifest: 891 passengers with
survival outcome, sex, age, ticket class, fare, family counts, and port of
embarkation. A versioned copy is committed at `data/samples/titanic.csv` so
training is reproducible and offline.

Columns encoding the target were removed before training. `alive` is `survived`
as yes/no text and is refused by an automatic guard, along with any feature that
predicts the target at or above 0.99 accuracy. Columns duplicating or derived
from others (`class`, `who`, `adult_male`, `embark_town`, `alone`) were dropped,
as was `deck`, which is missing for roughly three quarters of rows.

The data records one disaster in 1912. It is not a sample of any population that
exists now.\
"""

INTENDED_USE = """\
Teaching and demonstration. This model exists to exercise a training,
evaluation, and deployment pipeline end to end, and to show what honest
reporting of a weak model looks like.

It is a reasonable subject for discussing feature leakage, disaggregated
evaluation, and baseline comparison. It is not intended to be used to predict
anything about a living person.\
"""

OUT_OF_SCOPE_USES = """\
Any decision affecting a real individual. The model's dominant feature is sex,
and its second is ticket class. A system that assigns outcomes to people on
those grounds would be unlawful in most jurisdictions and unethical regardless
of legality.

Specifically out of scope: insurance pricing, risk scoring, triage, eligibility
decisions, or any transfer of these patterns to a modern maritime, aviation, or
emergency-response context. Evacuation norms in 1912 are not a model of how
people behave or should be treated now.\
"""

ETHICAL_CONSIDERATIONS = """\
**The model encodes a historical social order, not a causal one.** Sex and
ticket class predict survival here because of how the 1912 evacuation was
conducted -- lifeboat access, deck location, and the norms applied on the night.
The model has learned the consequences of those decisions. It has not learned
anything about who survives emergencies in general, and reading it that way
would launder a historical injustice as a finding.

**Using sex as a feature is deliberate and would not be acceptable in
deployment.** It is retained because the exercise is to study this dataset
honestly, and removing the strongest predictor while continuing to report high
accuracy would be misleading in a different direction. In any real system,
protected attributes require an explicit lawful basis, documented necessity, and
a fairness assessment before use. This model has none of those and should not be
deployed.

**The disaggregated results show concrete harm if it were.** The model predicts
death for every man in second and third class and survival for nearly every
woman in first and second. Deployed against people, it would issue categorical
verdicts by demographic group while presenting an aggregate accuracy that
conceals this entirely.

**Aggregate accuracy is the misleading number here.** Most of it comes from the
base rates of two attributes. The baseline comparison above is the honest
summary of what the model contributes.\
"""

ADDITIONAL_LIMITATIONS = """\
- The strongest predictor is a protected attribute, so the model's apparent skill
  is inseparable from a demographic split of the outcome.
- Age is missing for about 20% of rows and is median-imputed. Predictions for
  passengers with unknown age rest on a population median rather than anything
  about them.\
"""


def run(settings: AnalyticsSettings | None = None, model_name: str = "logistic_regression") -> Path:
    """Generate a model card for a stored model version and write it beside it."""
    resolved = settings or get_analytics_settings()
    registry = ModelRegistry(resolved.model_dir)

    pipeline, metadata = registry.load(model_name)

    frame = load_titanic(settings=resolved)
    features, target = build_features(frame, settings=resolved)
    data = split_data(features, target, settings=resolved)

    groups = evaluate_by_group(pipeline, data.features_test, data.target_test, GROUP_COLUMNS)
    baseline = compare_to_baseline(
        pipeline,
        data.features_train,
        data.target_train,
        data.features_test,
        data.target_test,
        GROUP_COLUMNS,
    )

    card = generate_card(
        metadata=metadata,
        groups=groups,
        baseline=baseline,
        dataset_description=DATASET_DESCRIPTION,
        intended_use=INTENDED_USE,
        out_of_scope_uses=OUT_OF_SCOPE_USES,
        ethical_considerations=ETHICAL_CONSIDERATIONS,
        additional_limitations=ADDITIONAL_LIMITATIONS,
    )

    destination = resolved.model_dir / model_name / metadata.version / CARD_FILENAME
    destination.write_text(card, encoding="utf-8")

    logger.info(
        "model_card_written", model=model_name, version=metadata.version, path=str(destination)
    )
    return destination


def main() -> None:
    """Console entry point: zepto-card."""
    configure_logging()
    destination = run()
    print(f"Model card written to {destination}")
