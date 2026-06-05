from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import TASK_TYPES

app = FastAPI(title="token-optimizer", version="0.1.0")


class RunRequest(BaseModel):
    task_type: str
    repo: str
    input: str
    model: Optional[str] = None


class TokenReportOut(BaseModel):
    baseline_tokens: int
    context_tokens: int
    input_tokens: int
    output_tokens: int
    savings_pct: float
    model_used: str


class RunResponse(BaseModel):
    task_type: str
    output: str
    token_report: TokenReportOut


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/run", response_model=RunResponse)
def run_task(req: RunRequest):
    if req.task_type not in TASK_TYPES:
        raise HTTPException(status_code=400, detail=f"task_type must be one of {TASK_TYPES}")

    from token_optimizer.engine import optimize

    result = optimize(req.task_type, req.repo, req.input, override_model=req.model)
    r = result.token_report
    return RunResponse(
        task_type=result.task_type,
        output=result.output,
        token_report=TokenReportOut(
            baseline_tokens=r.baseline_tokens,
            context_tokens=r.context_tokens,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            savings_pct=r.savings_pct,
            model_used=r.model_used,
        ),
    )
