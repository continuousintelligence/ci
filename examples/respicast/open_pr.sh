#!/usr/bin/env bash
# Open the pull request that submits ci-proto's forecasts for one round to RespiCast.
#
#   bash examples/respicast/open_pr.sh ROUND FOLDER
#
# ROUND is the round's date and FOLDER is submit.py's --output folder. The file goes to a new
# branch of FORK, made from its main once that is brought up to the hub's, and the pull request
# goes from there to HUB, whose checks merge a valid submission by themselves. Nothing happens if
# the round is already on the hub's main or has an open pull request. With DRY_RUN=1 it runs those
# checks and says what it would do. GH_TOKEN must be able to write to FORK and open pull requests
# on HUB.
set -euo pipefail

round=$1
folder=$2
[[ $round =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "Not a date: $round" >&2; exit 1; }

hub=${HUB_REPO:-european-modelling-hubs/RespiCast-SyndromicIndicators}
fork=${FORK_REPO:-Benjamin-Walker/RespiCast-SyndromicIndicators}
name=CI-ci_proto
path=model-output/$name/$round-$name.csv
branch=add-$name-$round
title="Add $name forecasts for $round"
body="Weekly forecasts from $name for the $(date -d "$round" '+%-d %B %Y') round."
[[ -s $folder/$path ]] || { echo "No submission at $folder/$path" >&2; exit 1; }

# Delete the fork's branches from earlier submissions once the hub has merged or closed their
# pull requests, so the fork keeps only main.
for old in $(gh api "repos/$fork/branches?per_page=100" --paginate \
    --jq ".[].name | select(startswith(\"add-$name-\"))"); do
    state=$(gh pr list -R "$hub" --head "$old" --state all --json state,headRepositoryOwner \
        --jq "[.[] | select(.headRepositoryOwner.login == \"${fork%%/*}\")][0].state // \"\"")
    if [[ $state == MERGED || $state == CLOSED ]]; then
        if [[ -n ${DRY_RUN:-} ]]; then
            echo "Dry run: would delete the branch $fork:$old (pull request ${state,,})."
        else
            gh api -X DELETE "repos/$fork/git/refs/heads/$old" --silent
            echo "Deleted the branch $fork:$old (pull request ${state,,})."
        fi
    fi
done

if gh api "repos/$hub/contents/$path?ref=main" --silent 2>/dev/null; then
    echo "The $round round is already on the hub's main branch; nothing to do."
    exit 0
fi
open=$(gh pr list -R "$hub" --head "$branch" --state open --json url,headRepositoryOwner \
    --jq "[.[] | select(.headRepositoryOwner.login == \"${fork%%/*}\")][0].url // \"\"")
if [[ -n $open ]]; then
    echo "The $round round already has an open pull request: $open"
    exit 0
fi
if [[ -n ${DRY_RUN:-} ]]; then
    echo "Dry run: would put $path on $fork:$branch and open \"$title\" on $hub."
    exit 0
fi

# Bring the fork's main up to the hub's, then branch from it. GitHub refuses the update when the
# hub has changed its own workflows and the token lacks the workflow scope; the fork's main is
# then older but still part of the hub's history, so the pull request holds only the new file.
if ! gh api -X POST "repos/$fork/merge-upstream" -f branch=main --silent; then
    echo "::warning::Could not update $fork's main from $hub; branching from it as it is."
fi
sha=$(gh api "repos/$fork/git/ref/heads/main" --jq .object.sha)
gh api -X DELETE "repos/$fork/git/refs/heads/$branch" --silent 2>/dev/null || true
gh api "repos/$fork/git/refs" -f ref="refs/heads/$branch" -f sha="$sha" --silent

# The file goes in through the contents API, as one commit on the new branch.
payload=$(mktemp)
trap 'rm -f "$payload"' EXIT
base64 -w0 "$folder/$path" | jq -Rs --arg message "$title" --arg branch "$branch" \
    --arg author "${AUTHOR_NAME:?}" --arg email "${AUTHOR_EMAIL:?}" \
    '{message: $message, branch: $branch, content: .,
      author: {name: $author, email: $email}, committer: {name: $author, email: $email}}' \
    > "$payload"
gh api -X PUT "repos/$fork/contents/$path" --input "$payload" --silent

url=$(gh api "repos/$hub/pulls" -f title="$title" -f body="$body" -f base=main \
    -f head="${fork%%/*}:$branch" --jq .html_url)
echo "Opened $url"
