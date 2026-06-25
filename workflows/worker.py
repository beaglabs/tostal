import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from activities.model_inference import run_inference
from activities.storage_ops import convert_file, read_from_icechunk, write_to_icechunk
from workflows import ClassifyWorkflow, KrigeWorkflow, SegmentWorkflow


async def main():
    host = os.environ.get("TEMPORAL_HOST", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")

    client = await Client.connect(host, namespace=namespace)

    worker = Worker(
        client,
        task_queue="tostal-task-queue",
        workflows=[ClassifyWorkflow, KrigeWorkflow, SegmentWorkflow],
        activities=[run_inference, read_from_icechunk, write_to_icechunk, convert_file],
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())