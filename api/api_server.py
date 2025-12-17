import os
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"


from contextlib import asynccontextmanager

import asyncio
import io
import os
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from fastapi.responses import StreamingResponse
import torchaudio

from api.utils import short_prompt
from api.compute_agent import GenerateArgs, MultimodelGenerateAgent
from typing import Literal, MutableMapping, Optional
import uuid
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    # special field to force not having extra arguments
    model_config = ConfigDict(extra="forbid")
    version: Literal["0.1"] = "0.1"
    token: str
    args: GenerateArgs = GenerateArgs()


class QueueStatusResponse(BaseModel):
    request_id: str
    status: str  # "queued", "processing", "completed", "failed"
    position: Optional[int] = None
    estimated_wait_seconds: Optional[float] = None
    result_available: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to handle startup and shutdown tasks.

    Args:
        app (FastAPI):
    """
    global request_queue
    request_queue = asyncio.Queue(maxsize=20)  # Initialize async queue
    queue_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_old_requests())
    logger.info("Background queue processor started")
    # Load the ML model
    compute_agent.load_model()
    yield
    # Clean up the ML models and release the resources
    queue_task.cancel()
    cleanup_task.cancel()


class ActiveRequest:
    def __init__(
        self,
        request_id: str,
        created_at: float,
        prompt: str,
        index: int,
        status: str = "queued",
        generation_time: Optional[float] = None,
        result: Optional[bytes] = None,
        error: Optional[str] = None,
        completed_at: Optional[float] = None,
    ):
        self.request_id = request_id
        self.status = status
        self.created_at = created_at
        self.prompt = prompt
        self.index = index
        self.generation_time = generation_time
        self.result = result
        self.error = error
        self.completed_at = completed_at
        self.event = asyncio.Event()


app = FastAPI(lifespan=lifespan)

# Request queue management (initialized in startup event)
request_queue: Optional[asyncio.Queue] = None
active_requests: MutableMapping[str, ActiveRequest] = {}  # Track request status
queue_lock = Lock()
requests_counter = 0  # Global request counter
requests_done = 0  # Global requests done counter


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Compute agent
compute_agent = MultimodelGenerateAgent(
    {
        "SFXflowmap": "flowmap",
    },
    device=device,
)


def run_generate(req: GenerateRequest):
    """Run the generation process.
    calls compute_agent.
    """
    # @TODO use a lock to prevent concurrent access in the compute_agent
    args = req.args
    logger.info(
        f"Starting generation for prompt: '{short_prompt(args.prompt)}' "
        f"[steps: {args.num_steps}, guidance: {args.cfg}]"
    )
    start_time = time.time()

    try:
        # @TODO when extending to different shapes and models, clear GPU cache and move models to cpu
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        result = compute_agent.generate(args)

        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        return result, generation_time
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(
            f"Error during  generation after {generation_time:.2f}s: {type(e).__name__}: {str(e)}"
        )
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def compress_audio(
    wav_tensor, sr, format="flac", bits_per_sample=16, backend="soundfile"
):
    buf = io.BytesIO()
    torchaudio.save(
        buf,
        wav_tensor,
        sample_rate=sr,
        format=format,
        bits_per_sample=bits_per_sample,
        backend=backend,
    )
    buf.seek(0)
    return buf


# Start background queue processor
async def process_queue():
    """Background task to process the request queue"""
    global requests_done
    while True:
        try:
            # Get next request from queue
            if request_queue is None:
                await asyncio.sleep(1)
                continue
            request_data = await request_queue.get()
            request_id = request_data["request_id"]
            request_data = request_data["data"]
            # Update status to processing
            with queue_lock:
                if request_id in active_requests:
                    active_requests[request_id].status = "processing"

            logger.info(f"Processing request {request_id}")

            try:
                # Process the generation request
                result, generation_time = await asyncio.to_thread(
                    run_generate,
                    request_data,
                )

                # buf = compress_audio(result["audio"], result["sample_rate"])
                buf = await asyncio.to_thread(
                    compress_audio,
                    result["audio"],
                    result["sample_rate"],
                )
                # Store result
                with queue_lock:
                    if request_id in active_requests:
                        current_request = active_requests[request_id]
                        current_request.result = buf.getvalue()
                        current_request.generation_time = generation_time
                        current_request.status = "completed"
                        current_request.completed_at = time.time()
                        current_request.event.set()  # Notify any waiting tasks
                logger.info(f"Request {request_id} completed in {generation_time:.2f}s")

            except Exception as e:
                logger.error(f"Error processing request {request_id}: {str(e)}")
                with queue_lock:
                    if request_id in active_requests:
                        current_request = active_requests[request_id]
                        current_request.status = "failed"
                        current_request.error = str(e)
                        current_request.completed_at = time.time()
                        current_request.event.set()  # Notify any waiting tasks

            finally:
                if request_queue is not None:
                    request_queue.task_done()
                # increment global request counter
                requests_done += 1

        except Exception as e:
            logger.error(f"Queue processor error: {str(e)}")
            await asyncio.sleep(1)


# Cleanup old completed requests
async def cleanup_old_requests():
    """Clean up old completed/failed requests to prevent memory buildup"""
    while True:
        try:
            current_time = time.time()
            with queue_lock:
                # Remove requests older than 10 minutes
                old_requests = [
                    req_id
                    for req_id, req_data in active_requests.items()
                    if req_data.completed_at
                    and req_data.status in ["completed", "failed"]
                    and req_data.completed_at < current_time - 600
                ]
                for req_id in old_requests:
                    del active_requests[req_id]

                if old_requests:
                    logger.info(f"Cleaned up {len(old_requests)} old requests")

        except Exception as e:
            logger.error(f"Cleanup task error: {str(e)}")

        await asyncio.sleep(300)  # Run every 5 minutes


@app.get("/ping")
def ping():
    """Get server status"""
    # @TODO check compute agent and billing agent
    return {"status": "ok"}


@app.get("/queue/status")
async def queue_status():
    """Get current queue status"""
    queue_size = request_queue.qsize() if request_queue else 0
    with queue_lock:
        active_count = len(
            [r for r in active_requests.values() if r.status == "processing"]
        )

    return {
        "queue_size": queue_size,
        "active_requests": active_count,
        "total_slots": request_queue.maxsize if request_queue else 20,
    }


@app.get("/queue/status/{request_id}")
async def get_request_status(request_id: str):
    """Get status of specific request"""
    with queue_lock:
        if request_id not in active_requests:
            raise HTTPException(status_code=404, detail="Request not found")

        request_info = active_requests[request_id]

        # Calculate position in queue if still queued
        position = None
        estimated_wait = None
        if request_info.status == "queued":
            # Estimate position and wait time
            position = request_info.index - requests_done
            if position < 0:
                position = 0  # Ensure position is not negative
            # Estimate wait time based on position
            estimated_wait = position * 2.0  # Estimate 2 seconds per request

        return QueueStatusResponse(
            request_id=request_id,
            status=request_info.status,
            position=position,
            estimated_wait_seconds=estimated_wait,
            result_available=request_info.status == "completed",
        )


@app.get("/queue/sse_updates/{request_id}")
async def stream_request_status(request_id: str):
    """Get status of specific request as an SSE stream"""
    with queue_lock:
        if request_id not in active_requests:
            raise HTTPException(status_code=404, detail="Request not found")
        status = active_requests[request_id].status

    async def event_stream(status):
        while status == "queued" or status == "processing":
            with queue_lock:
                request_info = active_requests[request_id]
                status = request_info.status
                position = request_info.index - requests_done
                if position < 0:
                    position = 0  # Ensure position is not negative
            if status == "queued":
                yield f"status: {status}, position: {position} \n\n"
            await asyncio.sleep(0.5)  # Poll every 0.5 second

    return StreamingResponse(event_stream(status), media_type="text/event-stream")


@app.post("/generate/queue")
async def generate_queue(req: GenerateRequest):
    global requests_counter
    """Queue a new generation request"""
    try:
        # Check if queue is full
        if request_queue is None or request_queue.full():
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Queue is full. Please try again later.",
            )

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        logger.info(
            f"Queuing generation request {request_id} for prompt: '{short_prompt(req.args.prompt)}'"
        )

        # Add to active requests tracking
        with queue_lock:
            active_requests[request_id] = ActiveRequest(
                request_id=request_id,
                created_at=time.time(),
                prompt=short_prompt(req.args.prompt, 100),
                index=requests_counter,
            )
            requests_counter += 1

        # Add request to queue
        request_data = {"request_id": request_id, "data": req}

        await request_queue.put(request_data)

        # Return request ID and status
        return {
            "request_id": request_id,
            "status": "queued",
            "message": "Request queued for processing",
            "queue_position": request_queue.qsize() if request_queue else 0,
            "estimated_wait_seconds": (request_queue.qsize() if request_queue else 0)
            * 3.0,
            "status_url": f"/queue/status/{request_id}",
            "result_url": f"/result/{request_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to queue request: {str(e)}"
        )


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """Get the result of a completed generation request"""
    with queue_lock:
        if request_id not in active_requests:
            raise HTTPException(status_code=404, detail="Request not found")

        request_info = active_requests[request_id]

        if request_info.status == "completed":
            # Copy the data to use outside the lock
            audio_data = request_info.result
            if audio_data is None:
                raise HTTPException(
                    status_code=500, detail="No audio data found for this request."
                )
            buf = io.BytesIO(audio_data)

        elif request_info.status == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {request_info.error or 'Unknown error'}",
            )
        else:
            raise HTTPException(
                status_code=202,
                detail=f"Request is {request_info.status}. Check status endpoint for updates.",
            )

    return StreamingResponse(buf, media_type="audio/flac")


@app.get("/result/await/{request_id}")
async def await_result(request_id: str):
    """Get the result of a generation request. if it's not completed waits completion and returns the result."""
    with queue_lock:
        if request_id not in active_requests:
            raise HTTPException(status_code=404, detail="Request not found")

        request_info = active_requests[request_id]
    try:
        # this has to be outside the lock, but fastapi should only have a single thread
        await request_info.event.wait()  # Wait for completion
        logger.info(
            f"get_result request finished, Prompt: '{short_prompt(request_info.prompt)}'"
        )
        audio_data = request_info.result
        if audio_data is None:
            raise HTTPException(
                status_code=500, detail="No audio data found for this request."
            )
        buf = io.BytesIO(audio_data)
        return StreamingResponse(buf, media_type="audio/flac")

    except Exception as e:
        logger.error(
            f"Error in synchronous generate endpoint: {type(e).__name__}: {str(e)}"
        )
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {type(e).__name__}: {str(e)}",
        )


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Synchronous generation endpoint, adds request to queue and waits for resutls, returns the audio"""
    logger.info(
        f"Received synchronous generation request for prompt: '{short_prompt(req.args.prompt)}'"
    )
    queue_result = await generate_queue(req)
    try:
        request_id = queue_result["request_id"]
        with queue_lock:
            if request_id not in active_requests:
                raise HTTPException(
                    status_code=500, detail="Request queued but not found in queue."
                )
            request_info = active_requests[request_id]
        logger.info(
            f"Synchronous queued and waiting for results, position={queue_result['queue_position']}, Prompt: '{short_prompt(req.args.prompt)}'"
        )
        # this has to be outside the lock, but fastapi should only have a single thread
        await request_info.event.wait()  # Wait for completion
        logger.info(
            f"Synchronous request finished, position was {queue_result['queue_position']}, Prompt: '{short_prompt(req.args.prompt)}'"
        )
        audio_data = request_info.result
        if audio_data is None:
            raise HTTPException(
                status_code=500, detail="No audio data found for this request."
            )
        buf = io.BytesIO(audio_data)
        return StreamingResponse(buf, media_type="audio/flac")

    except Exception as e:
        logger.error(
            f"Error in synchronous generate endpoint: {type(e).__name__}: {str(e)}"
        )
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {type(e).__name__}: {str(e)}",
        )


@app.post("/generate/priority")
async def generate_priority(req: GenerateRequest):
    """Synchronous without queue generation endpoint"""
    try:
        result, generation_time = await asyncio.to_thread(
            run_generate,
            req,
        )
        print("result", result)
        # buf = compress_audio(result["audio"], result["sample_rate"])
        buf = await asyncio.to_thread(
            compress_audio,
            result["audio"],
            result["sample_rate"],
        )

        logger.info(
            f"Synchronous generation completed in {generation_time:.2f} seconds"
        )
        return StreamingResponse(buf, media_type="audio/flac")

    except Exception as e:
        logger.error(
            f"Error in synchronous generate endpoint: {type(e).__name__}: {str(e)}"
        )
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {type(e).__name__}: {str(e)}",
        )
