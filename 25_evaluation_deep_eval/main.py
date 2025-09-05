from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GeminiModel
import os

from dotenv import load_dotenv

load_dotenv("./.env")

# Define Evaluation Model

model = GeminiModel(
    model_name="gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Test Correctness
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    model=model,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        # "You should also heavily penalize omission of detail",
        # "Vague language, or contradicting OPINIONS, are OK"
    ]
)

gt_answer = "Rome is the capital of Italy."
pred_answer = "Rome"

test_case_correctness = LLMTestCase(
    input="What is the capital of Italy?",
    expected_output=gt_answer,
    actual_output=pred_answer
)

# correctness_metric.measure(test_case_correctness)
# print(correctness_metric.score, correctness_metric.reason)

# Test Faithfulness
question = "What is 3+3?"
context = ["6"]
generated_answer = "6"

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=model,
)

test_case_faithfulness = LLMTestCase(
    input=question,
    actual_output=generated_answer,
    retrieval_context=context
)

# faithfulness_metric.measure(test_case=test_case_faithfulness)
# print(faithfulness_metric.score)
# print(faithfulness_metric.reason)

# Test contextual relevancy
actual_output = "then go somewhere else."
retrieval_context = ["this is a test context", "mike is a cat", "if the shoes don't fit, then go somewhere else."]
gt_answer = "if the shoes don't fit, then go somewhere else."

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model=model,
    include_reason=True
)

relevance_test_case = LLMTestCase(
    input="what if these shoes don't fit?",
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output=gt_answer
)

# relevance_metric.measure(relevance_test_case)
# print(relevance_metric.score)
# print(relevance_metric.reason)

# Test two different cases together with several metrics together
new_test_case = LLMTestCase(
    input="what is the capital of Spain?",
    expected_output="Madrid is the capital of Spain.",
    actual_output="MadriD.",
    retrieval_context=["Madrid is the capital of Spain."]
)

evaluate(
    test_cases=[relevance_test_case, new_test_case],
    metrics=[correctness_metric, faithfulness_metric, relevance_metric]
)