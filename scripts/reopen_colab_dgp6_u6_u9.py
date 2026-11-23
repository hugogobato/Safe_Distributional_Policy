"""Reopen the 12 saved DGP 6 Colab Drive URLs (strictly chrome u/6..u/9),
check whether remote VMs are still running or need Run All, and run a 5-min health check.
"""

import asyncio
import importlib.util
import json
import pathlib
import random
import time

COLAB_FLEET_PATH = pathlib.Path(
    "/home/hugo/.local/share/agy-profiles/gemini/home/.gemini/config/skills/colab-fleet-runner/scripts/colab_fleet.py"
)
REPORT_IN = pathlib.Path(
    "/home/hugo/Stuff/Research/Safe_Distributional_Policy/colab_notebooks/phase4_v2_dgp6/health_report.json"
)
REPORT_OUT = pathlib.Path(
    "/home/hugo/Stuff/Research/Safe_Distributional_Policy/colab_notebooks/phase4_v2_dgp6/health_report_reopened.json"
)
ALLOWED_ACCOUNTS = (6, 7, 8, 9)


def _load_cf():
    spec = importlib.util.spec_from_file_location("colab_fleet", COLAB_FLEET_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def main():
    cf = _load_cf()
    from playwright.async_api import async_playwright

    prev = json.loads(REPORT_IN.read_text(encoding="utf-8"))
    assert len(prev) == 12
    for item in prev:
        assert item["account_idx"] in ALLOWED_ACCOUNTS, f"Unauthorized account {item['account_idx']}"

    cdp_url = cf.ensure_browser_running("chrome", resync_auth=True)
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        ctx = browser.contexts[0]
        sem = asyncio.Semaphore(2)

        async def open_one(idx: int, item: dict):
            u = item["account_idx"]
            url = item["drive_url"]
            assert f"authuser={u}" in url
            async with sem:
                page = await ctx.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await asyncio.sleep(6.0)
                await cf.handle_colab_dialogs(page, allow_terminate_others=False)
                info = await cf.inspect_tab_status(page)
                verdict = cf.classify_health(info)
                action = "OBSERVED"
                if verdict["health"] not in ("RUNNING_OK", "COMPLETED_OK"):
                    try:
                        await cf.trigger_run_all(page, allow_terminate_others=False)
                        action = "RETRIGGERED_RUN_ALL"
                    except Exception as exc:
                        action = f"RETRIGGER_ERR:{exc}"
                print(
                    f"  [{idx}/12] chrome u/{u} slot {item['slot_idx']} ({item['notebook']}): "
                    f"initial={verdict['health']} ({verdict['reason']}) -> {action}",
                    flush=True,
                )
                return {**item, "page": page, "action": action}

        tasks = []
        for idx, item in enumerate(prev, 1):
            tasks.append(asyncio.create_task(open_one(idx, item)))
            if idx < len(prev):
                await asyncio.sleep(random.uniform(1.8, 2.6))
        records = list(await asyncio.gather(*tasks))

        # Close initial blank page
        for pg in list(ctx.pages):
            if pg.url in ("about:blank", "chrome://new-tab-page/"):
                try:
                    await pg.close()
                except Exception:
                    pass

        print("\n[WAIT] Waiting 300s (5.0 min) before health check...", flush=True)
        t0 = time.time()
        while True:
            rem = 300 - (time.time() - t0)
            if rem <= 0:
                break
            await asyncio.sleep(min(30.0, rem))
            for rec in records:
                try:
                    await cf.handle_colab_dialogs(rec["page"], allow_terminate_others=False)
                except Exception:
                    pass
            print(f"  [WAIT] {time.time() - t0:.0f}s / 300s elapsed...", flush=True)

        print("\n[HEALTH CHECK] Inspecting all 12 tabs (chrome u/6..u/9)...", flush=True)
        out = []
        ok = 0
        for rec in records:
            pg = rec["page"]
            await cf.handle_colab_dialogs(pg, allow_terminate_others=False)
            info = await cf.inspect_tab_status(pg)
            verdict = cf.classify_health(info)
            entry = {
                "browser": "chrome",
                "account_idx": rec["account_idx"],
                "slot_idx": rec["slot_idx"],
                "notebook": rec["notebook"],
                "drive_url": rec["drive_url"],
                "action": rec["action"],
                "health": verdict["health"],
                "reason": verdict["reason"],
                "last_log": info.get("lastLog", ""),
                "run_states": info.get("runStates", []),
            }
            if verdict["health"] in ("RUNNING_OK", "COMPLETED_OK"):
                ok += 1
            out.append(entry)
            print(
                f"  - chrome u/{entry['account_idx']} slot {entry['slot_idx']} "
                f"({entry['notebook']}): [{entry['health']}] {entry['reason']}",
                flush=True,
            )
        REPORT_OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\n[SUMMARY] Healthy/Completed: {ok}/{len(records)} | Report: {REPORT_OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
