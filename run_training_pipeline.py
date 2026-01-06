import os
from mlproject.pipelines.training_pipeline import pipeline

pipeline.upsert(role_arn=os.environ["SAGEMAKER_ROLE_ARN"])
pipeline.start()

pipeline.start()
