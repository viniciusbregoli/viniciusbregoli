# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dateutil"]
# ///
"""Fetch GitHub stats and write the neofetch-style profile card (dark and light SVG).

    GITHUB_TOKEN=... uv run card/build.py

Repos, stars and lines of code come from public repos only. Commits count whatever
the token can see: the Actions token sees public commits, a personal token with
access to private repos adds those.
"""
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import httpx
from dateutil.relativedelta import relativedelta

LOGIN = "viniciusbregoli"
HERE = Path(__file__).parent
LOC_CACHE = HERE / "loc_cache.json"

WIDTH = 60  # characters in every info line, so the dot leaders line up
FONT_SIZE = 16
LINE_HEIGHT = 20
CHAR_WIDTH = 9.65  # Consolas-ish advance at 16px; only used to size the canvas
# The portrait uses a smaller font so it gets about three times as many characters
# as the info panel would allow; at 16px the face is too few cells to read.
ART_FONT_SIZE = 10
ART_LINE_HEIGHT = 12
ART_CHAR_WIDTH = CHAR_WIDTH * ART_FONT_SIZE / FONT_SIZE
PAD = 20
GAP = 40

THEMES = {
    "dark": {
        "bg": "#161b22",
        "border": "#30363d",
        "text": "#c9d1d9",
        "key": "#ffa657",
        "value": "#a5d6ff",
        "dots": "#616e7f",
        "add": "#3fb950",
        "del": "#f85149",
    },
    "light": {
        "bg": "#f6f8fa",
        "border": "#d0d7de",
        "text": "#24292f",
        "key": "#953800",
        "value": "#0a3069",
        "dots": "#8c959f",
        "add": "#1a7f37",
        "del": "#cf222e",
    },
}

# Static lines of the card. Each entry is (key, value); None is a blank spacer line.
ABOUT = [
    ("OS", "macOS, iOS, Debian"),
    ("Uptime", None),  # filled in from the account age
    ("Host", "Kinebot"),
    ("Kernel", "AI Engineer and Team Lead"),
    ("Shell", "fish"),
    ("Location", "Curitiba, Brazil"),
    None,
    ("Languages.Programming", "Python, TypeScript, Swift, Rust"),
    ("Languages.Computer", "SQL, HTML, CSS, LaTeX, YAML"),
    ("Languages.Real", "Portuguese, English, German"),
    None,
    ("Hobbies.Software", "macOS and iOS apps, tiling WMs"),
    ("Hobbies.Real", "Guitar, my band, learning German"),
]
CONTACT = [
    ("Website", "viniciusbregoli.dev"),
    ("LinkedIn", "viniciusbregoli"),
    ("GitHub", "viniciusbregoli"),
]


# ---------------------------------------------------------------- data


def client() -> httpx.Client:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN (or GH_TOKEN) to call the GitHub API.")
    return httpx.Client(
        base_url="https://api.github.com",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=30,
    )


def graphql(gh: httpx.Client, query: str, **variables) -> dict:
    r = gh.post("/graphql", json={"query": query, "variables": variables})
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        sys.exit(f"GraphQL error: {body['errors']}")
    return body["data"]


def profile_stats(gh: httpx.Client) -> dict:
    query = """
    query($login: String!, $after: String) {
      user(login: $login) {
        createdAt
        followers { totalCount }
        repositories(first: 100, after: $after, ownerAffiliations: OWNER, privacy: PUBLIC) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes { name isFork stargazerCount }
        }
      }
    }
    """
    repos, after = [], None
    while True:
        user = graphql(gh, query, login=LOGIN, after=after)["user"]
        page = user["repositories"]
        repos += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    return {
        "created_at": datetime.fromisoformat(user["createdAt"]),
        "followers": user["followers"]["totalCount"],
        "repos": page["totalCount"],
        "stars": sum(r["stargazerCount"] for r in repos),
        "own_repos": [r["name"] for r in repos if not r["isFork"]],
    }


def contributions(gh: httpx.Client, since: datetime) -> tuple[int, int]:
    """Count lifetime commits and the repos I contributed to.

    contributionsCollection only spans one year per call, so walk year by year.
    repositoriesContributedTo would be one call, but it only covers recent activity.
    """
    query = """
    query($login: String!, $from: DateTime!, $to: DateTime!) {
      user(login: $login) {
        contributionsCollection(from: $from, to: $to) {
          totalCommitContributions
          commitContributionsByRepository(maxRepositories: 100) { repository { nameWithOwner } }
          pullRequestContributionsByRepository(maxRepositories: 100) { repository { nameWithOwner } }
          pullRequestReviewContributionsByRepository(maxRepositories: 100) { repository { nameWithOwner } }
          issueContributionsByRepository(maxRepositories: 100) { repository { nameWithOwner } }
        }
      }
    }
    """
    now = datetime.now(UTC)
    commits, repos, start = 0, set(), since
    while start < now:
        end = min(start + relativedelta(years=1), now)
        data = graphql(gh, query, login=LOGIN, **{"from": start.isoformat(), "to": end.isoformat()})
        c = data["user"]["contributionsCollection"]
        commits += c["totalCommitContributions"]
        for key, groups in c.items():
            if key.endswith("ByRepository"):
                repos.update(g["repository"]["nameWithOwner"] for g in groups)
        start = end
    return commits, len(repos)


def lines_of_code(gh: httpx.Client, repos: list[str]) -> tuple[int, int]:
    """Sum my additions and deletions across my public repos.

    GitHub computes contributor stats lazily and answers 202 until they're ready,
    so retry a few times and fall back to the last known numbers for that repo.
    """
    cache = json.loads(LOC_CACHE.read_text()) if LOC_CACHE.exists() else {}
    for repo in repos:
        for _ in range(6):
            r = gh.get(f"/repos/{LOGIN}/{repo}/stats/contributors")
            if r.status_code != 202:
                break
            time.sleep(5)
        if r.status_code == 200:
            mine = [c for c in r.json() if (c.get("author") or {}).get("login") == LOGIN]
            weeks = mine[0]["weeks"] if mine else []
            cache[repo] = [sum(w["a"] for w in weeks), sum(w["d"] for w in weeks)]
        elif r.status_code == 204:  # empty repo
            cache[repo] = [0, 0]
        else:
            print(f"loc: {repo} answered {r.status_code}, keeping cached value", file=sys.stderr)
    cache = {k: v for k, v in cache.items() if k in repos}
    LOC_CACHE.write_text(json.dumps(dict(sorted(cache.items())), indent=2) + "\n")
    return sum(a for a, _ in cache.values()), sum(d for _, d in cache.values())


def uptime(since: datetime) -> str:
    d = relativedelta(datetime.now(UTC), since)
    plural = lambda n, unit: f"{n} {unit}{'' if n == 1 else 's'}"
    return f"{plural(d.years, 'year')}, {plural(d.months, 'month')}, {plural(d.days, 'day')}"


# ---------------------------------------------------------------- rendering
# A line is a list of (css class, text) spans; the SVG writes them as tspans.

Span = tuple[str, str]


def kv(key: str, value: str, width: int = WIDTH) -> list[Span]:
    dots = width - len(". ") - len(key) - len(": ") - len(value) - 1
    return [
        ("dots", ". "),
        ("key", key),
        ("text", ":"),
        ("dots", " " + "." * max(dots, 1) + " "),
        ("value", value),
    ]


def header(title: str, lead: str = "") -> list[Span]:
    rest = WIDTH - len(lead) - len(title) - 1
    return [("text", lead + title + " "), ("text", "─" * rest)]


def stat_pair(left: tuple[str, str], right: tuple[str, str]) -> list[Span]:
    # Two key/value pairs on one line, split at a fixed column like neofetch's table.
    left_width = 35
    right_width = WIDTH - left_width - len(" | ")
    first = kv(*left, width=left_width)
    second = kv(*right, width=right_width + len(". "))[1:]
    return first + [("text", " | ")] + second


def build_lines(s: dict) -> list[list[Span]]:
    lines: list[list[Span]] = [header("vinicius@bregoli")]
    for item in ABOUT:
        if item is None:
            lines.append([("dots", ". ")])
            continue
        key, value = item
        lines.append(kv(key, uptime(s["created_at"]) if key == "Uptime" else value))
    lines.append([])
    lines.append(header("Contact", lead="- "))
    lines += [kv(k, v) for k, v in CONTACT]
    lines.append([])
    lines.append(header("GitHub Stats", lead="- "))
    lines.append(stat_pair(("Repos", f"{s['repos']} {{Contributed: {s['contributed']}}}"), ("Stars", f"{s['stars']:,}")))
    lines.append(stat_pair(("Commits", f"{s['commits']:,}"), ("Followers", f"{s['followers']:,}")))

    add, rem = s["loc"]
    tail = [("text", " ( "), ("add", f"{add:,}++"), ("text", ", "), ("del", f"{rem:,}--"), ("text", " )")]
    tail_len = sum(len(t) for _, t in tail)
    loc = kv("Lines of Code on GitHub", f"{add - rem:,}", width=WIDTH - tail_len)
    lines.append(loc + tail)
    return lines


def svg(theme: dict, ascii_art: list[str], lines: list[list[Span]]) -> str:
    art_cols = max(len(row) for row in ascii_art)
    art_height = len(ascii_art) * ART_LINE_HEIGHT
    info_height = len(lines) * LINE_HEIGHT
    content = max(art_height, info_height)
    info_x = PAD + art_cols * ART_CHAR_WIDTH + GAP
    width = round(info_x + WIDTH * CHAR_WIDTH + PAD)
    height = PAD * 2 + content
    # y is a text baseline, so start each block one line below its top edge.
    art_top = PAD + (content - art_height) // 2 + ART_LINE_HEIGHT
    info_top = PAD + (content - info_height) // 2 + LINE_HEIGHT

    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="ConsolasFallback,Consolas,Menlo,\'DejaVu Sans Mono\',monospace" '
            f'font-size="{FONT_SIZE}px">'
        ),
        "<style>",
        (
            f".text{{fill:{theme['text']}}} .key{{fill:{theme['key']}}} .value{{fill:{theme['value']}}} "
            f".dots{{fill:{theme['dots']}}} .add{{fill:{theme['add']}}} .del{{fill:{theme['del']}}}"
        ),
        "text,tspan{white-space:pre}",
        "</style>",
        f'<rect width="{width}" height="{height}" rx="15" fill="{theme["bg"]}" stroke="{theme["border"]}"/>',
        f'<text x="{PAD}" y="{art_top}" class="text" font-size="{ART_FONT_SIZE}px">',
    ]
    for i, row in enumerate(ascii_art):
        out.append(f'<tspan x="{PAD}" y="{art_top + i * ART_LINE_HEIGHT}">{escape(row)}</tspan>')
    out.append("</text>")
    out.append(f'<text x="{info_x:.0f}" y="{info_top}">')
    for i, spans in enumerate(lines):
        y = info_top + i * LINE_HEIGHT
        if not spans:
            continue
        # One XML line per card line: with white-space:pre, a newline between
        # tspans would render as an extra space and break the dot alignment.
        (cls, txt), *rest = spans
        line = f'<tspan x="{info_x:.0f}" y="{y}" class="{cls}">{escape(txt)}</tspan>'
        line += "".join(f'<tspan class="{c}">{escape(t)}</tspan>' for c, t in rest)
        out.append(line)
    out.append("</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    with client() as gh:
        s = profile_stats(gh)
        s["commits"], s["contributed"] = contributions(gh, s["created_at"])
        s["loc"] = lines_of_code(gh, s["own_repos"])

    lines = build_lines(s)
    for name, theme in THEMES.items():
        art = (HERE / f"ascii_{name}.txt").read_text().strip("\n").splitlines()
        (HERE / f"{name}_mode.svg").write_text(svg(theme, art, lines))
    summary = {k: v for k, v in s.items() if k not in ("own_repos", "created_at")}
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
