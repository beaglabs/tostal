import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.model_inference import run_inference
    from activities.storage_ops import read_from_icechunk, write_to_icechunk


@workflow.defn
class ClassifyWorkflow:
    @workflow.run
    async def run(self, params: dict) -> dict:
        customer_id = params["customer_id"]
        file_ids = params["file_ids"]
        task = params["task"]
        parameters = params.get("parameters", {})

        arrays = await workflow.execute_activity(
            read_from_icechunk,
            {"customer_id": customer_id, "file_ids": file_ids},
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        result = await workflow.execute_activity(
            run_inference,
            {"task": task, "model_type": "classify", "data": arrays, "parameters": parameters},
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        output_uri = await workflow.execute_activity(
            write_to_icechunk,
            {"customer_id": customer_id, "result": result, "prefix": f"results/classify_{task}"},
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return output_uri


@workflow.defn
class KrigeWorkflow:
    @workflow.run
    async def run(self, params: dict) -> dict:
        customer_id = params["customer_id"]
        file_ids = params["observations"]["file_ids"]
        variables = params["observations"]["variables"]
        grid = params["grid"]
        method = params.get("method", "murmurative")

        arrays = await workflow.execute_activity(
            read_from_icechunk,
            {"customer_id": customer_id, "file_ids": file_ids},
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        result = await workflow.execute_activity(
            run_inference,
            {
                "task": "kriging",
                "model_type": "krige",
                "data": arrays,
                "parameters": {"variables": variables, "grid": grid, "method": method},
            },
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        output_uri = await workflow.execute_activity(
            write_to_icechunk,
            {"customer_id": customer_id, "result": result, "prefix": "results/krige"},
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return output_uri


@workflow.defn
class SegmentWorkflow:
    @workflow.run
    async def run(self, params: dict) -> dict:
        customer_id = params["customer_id"]
        file_ids = params["file_ids"]
        task = params["task"]
        parameters = params.get("parameters", {})

        arrays = await workflow.execute_activity(
            read_from_icechunk,
            {"customer_id": customer_id, "file_ids": file_ids},
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        result = await workflow.execute_activity(
            run_inference,
            {"task": task, "model_type": "segment", "data": arrays, "parameters": parameters},
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        output_uri = await workflow.execute_activity(
            write_to_icechunk,
            {"customer_id": customer_id, "result": result, "prefix": f"results/segment_{task}"},
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return output_uri