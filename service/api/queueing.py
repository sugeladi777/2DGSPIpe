from redis import Redis
from rq import Queue

from service.api.config import settings
from service.worker import run_pipeline_job


def get_queue() -> Queue:
    redis = Redis.from_url(settings.redis_url)
    return Queue(name=settings.queue_name, connection=redis)


def enqueue_job(job_id: str) -> str:
    queue = get_queue()
    rq_job = queue.enqueue(
        run_pipeline_job,
        job_id,
        job_timeout=settings.job_timeout_sec,
        result_ttl=24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )
    return rq_job.id
