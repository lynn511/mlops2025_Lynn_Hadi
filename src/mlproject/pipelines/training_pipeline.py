# ========================
# Imports
# ========================
import os
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

# ========================
# AWS Config
# ========================
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
REGION = os.environ.get("AWS_REGION", "eu-north-1")
BUCKET = "lynn-hadi-taxi-mlops"

session = PipelineSession()

# ========================
# Preprocessing Processor
# ========================
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=session,
)

# ========================
# Preprocessing Step
# ========================
preprocess_step = ProcessingStep(
    name="Preprocess",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{BUCKET}/data/raw/",
            destination="/opt/ml/processing/input",
        ),
        ProcessingInput(
            source="src",
            destination="/opt/ml/processing/code/src",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="processed",
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/data/processed",
        )
    ],
    code="scripts/preprocess.py",
    job_arguments=[
        "--train_path", "/opt/ml/processing/input/train.csv",
        "--test_path", "/opt/ml/processing/input/test.csv",
        "--output_train", "/opt/ml/processing/output/train.csv",
        "--output_test", "/opt/ml/processing/output/test.csv",
    ],
)

# ========================
# Training Step (Linear only)
# ========================
estimator = SKLearn(
    entry_point="scripts/train.py",
    source_dir=".",
    dependencies=["src"],
    role=ROLE,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)




train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{BUCKET}/data/processed/train.csv",
            content_type="text/csv",
        )
    },
)

# ========================
# Pipeline Definition
# ========================
pipeline = Pipeline(
    name="TaxiTrainingPipeline",
    steps=[preprocess_step, train_step],
    sagemaker_session=session,
)

