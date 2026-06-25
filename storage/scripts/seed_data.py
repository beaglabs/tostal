"""Seed the database with test data for local development."""

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from app.models.db import Customer, File, Notebook, Job
from app.config import get_settings

settings = get_settings()


async def seed():
    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        # Check if data already exists
        result = await session.execute(select(Customer))
        if result.scalar_one_or_none():
            print("Database already seeded.")
            return

        customer = Customer(
            id=uuid.uuid4(),
            email="demo@tostal.com",
            name="Demo Customer",
            azure_container_name=f"container-demo-{uuid.uuid4().hex[:8]}",
            icechunk_store_uri=f"icechunk://azure://dev/demo",
            subscription_status="active",
        )
        session.add(customer)
        await session.flush()

        f1 = File(
            id=uuid.uuid4(),
            customer_id=customer.id,
            display_id="ing_demo001",
            filename="survey_2024.sgy",
            file_format="sgy",
            domain="geology",
            subdirectory="segy",
            icechunk_uri=f"icechunk://azure://dev/{customer.azure_container_name}/geology/segy/survey_2024",
            shape=[1024, 512, 256],
            dtype="float32",
            size_bytes=536870912,
            ingestion_status="completed",
        )
        f2 = File(
            id=uuid.uuid4(),
            customer_id=customer.id,
            display_id="ing_demo002",
            filename="well_b17.las",
            file_format="las",
            domain="geology",
            subdirectory="las",
            icechunk_uri=f"icechunk://azure://dev/{customer.azure_container_name}/geology/las/well_b17",
            shape=[15000, 8],
            dtype="float32",
            size_bytes=960000,
            ingestion_status="completed",
        )
        session.add_all([f1, f2])

        notebook = Notebook(
            id=uuid.uuid4(),
            customer_id=customer.id,
            display_id="nb_demo001",
            name="B-17 Basin Analysis",
            description="Facies mapping and kriging for the B-17 well field",
            icechunk_state_uri=f"icechunk://azure://dev/{customer.azure_container_name}/notebooks/nb_demo001/state.json",
            cell_count=2,
        )
        session.add(notebook)
        await session.flush()

        job = Job(
            id=uuid.uuid4(),
            customer_id=customer.id,
            notebook_id=notebook.id,
            display_id="cls_demo001",
            job_type="classify",
            task="facies-map",
            status="completed",
            input_file_ids=[str(f2.id)],
            parameters={"depth_range": [1200, 3500]},
            result_icechunk_uri=f"icechunk://azure://dev/{customer.azure_container_name}/geology/results/facies_map.zarr",
            result_shape=[4600, 1],
            classes=["sandstone", "shale", "limestone", "dolomite"],
            confidence=0.89,
        )
        session.add(job)
        await session.commit()

        print("Database seeded with demo data.")
        print(f"  Customer ID: {customer.id}")
        print(f"  Files: {f1.display_id}, {f2.display_id}")
        print(f"  Notebook: {notebook.display_id}")
        print(f"  Job: {job.display_id}")


if __name__ == "__main__":
    asyncio.run(seed())