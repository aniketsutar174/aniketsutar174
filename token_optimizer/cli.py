import click

from config import TASK_TYPES


@click.group()
def cli() -> None:
    """Token-efficient RAG optimizer for agentic dev tasks."""


@cli.command("index")
@click.option("--repo", required=True, help="Logical repo name (identifier in pgvector)")
@click.option("--root", required=True, type=click.Path(exists=True), help="Directory to index")
@click.option("--reindex", is_flag=True, default=False, help="Remove existing chunks first")
def cmd_index(repo: str, root: str, reindex: bool) -> None:
    """Index a codebase into pgvector."""
    from token_optimizer.retrieval.indexer import index_repo

    click.echo(f"Indexing '{root}' → repo '{repo}' ...")
    stats = index_repo(repo, root, reindex=reindex)
    click.echo(
        f"Done: {stats['files']} files  |  {stats['chunks']:,} chunks  "
        f"|  ~{stats['total_tokens']:,} tokens"
    )


@cli.command("run")
@click.argument("task_type", type=click.Choice(TASK_TYPES))
@click.option("--repo", required=True, help="Repo name to query against")
@click.option("--input", "user_input", required=True, help="Task input / question")
@click.option("--model", default=None, help="Override model (skips router)")
def cmd_run(task_type: str, repo: str, user_input: str, model: str) -> None:
    """Run a task against an indexed repo."""
    from token_optimizer.engine import optimize

    click.echo(f"Running [{task_type}] against repo '{repo}' ...")
    result = optimize(task_type, repo, user_input, override_model=model)

    click.echo("\n--- OUTPUT ---")
    click.echo(result.output)

    r = result.token_report
    click.echo("\n--- TOKEN REPORT ---")
    click.echo(f"  Baseline (full repo) : {r.baseline_tokens:>8,} tokens")
    click.echo(f"  Context sent         : {r.context_tokens:>8,} tokens")
    click.echo(f"  Input savings        : {r.savings_pct:>7.1f}%")
    click.echo(f"  Billed input/output  : {r.input_tokens:,} / {r.output_tokens:,}")
    click.echo(f"  Model                : {r.model_used}")


@cli.command("eval")
@click.option("--repo", required=True, help="Repo name to use for eval runs")
@click.option(
    "--golden",
    default="eval/golden.jsonl",
    type=click.Path(),
    help="Path to golden.jsonl",
)
def cmd_eval(repo: str, golden: str) -> None:
    """Run the evaluation harness and print a summary table."""
    from eval.harness import run_harness

    run_harness(repo=repo, golden_path=golden)
