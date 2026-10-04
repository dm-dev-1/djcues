"""`djcues loopout` -- re-place existing Loop Out cues with the
drums-no-vocals search (see loop_out.py). Registered from cli.py like the
tags group. Writes only Loop Out cues; Rekordbox must be closed and
master.db is backed up first. --dry-run lists what would move."""

from __future__ import annotations

import logging

import click

from djcues.db import get_db


@click.command("loopout")
@click.option("--under", default=None, type=click.Path(file_okay=False),
              help="Only tracks whose file is inside this folder.")
@click.option("--dry-run", is_flag=True, help="List what would move without writing.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def loopout(under, dry_run, yes):
    """Move existing Loop Out cues to the best drums-no-vocals stretch of
    each track's outro. Loops you moved or resized yourself are left alone."""
    from djcues.writer import PlaylistWriteError, apply_loop_out_realignment, plan_loop_out_realignment

    logging.getLogger("djcues.db").setLevel(logging.ERROR)
    db = get_db()
    plan = plan_loop_out_realignment(db, under=under)
    changes = plan["changes"]
    tracks = sorted({c["title"] for c in changes})
    hot = [c for c in changes if c["hot"]]
    eight = sum(1 for c in hot if c["bars"] == 8)
    click.echo(f"{len(tracks)} track(s) would get a new Loop Out ({eight} x 8 bars, {len(hot) - eight} x 4 bars).")
    for reason, titles in sorted(plan["skipped"].items()):
        click.echo(f"  left alone -- {reason}: {len(titles)}")
    for c in hot[:25]:
        click.echo(f"    {c['title'][:42]:42s} {c['old_in'] / 1000:7.1f}s -> {c['new_in'] / 1000:7.1f}s  ({c['bars']} bars)")
    if len(hot) > 25:
        click.echo(f"    ... and {len(hot) - 25} more")
    if dry_run or not changes:
        click.echo("Dry run -- nothing written." if dry_run else "Nothing to change.")
        return
    if not yes and not click.confirm("\nMove these Loop Out cues? (Rekordbox must be closed; master.db is backed up first)"):
        click.echo("Aborted.")
        return
    try:
        n = apply_loop_out_realignment(changes, db=db)
    except PlaylistWriteError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    click.echo(f"Moved {n} cue(s) on {len(tracks)} track(s).")
