#!/bin/bash

# Delete ALL issues (open and closed) from the repository
# Usage: ./scripts/delete_all_issues.sh

REPO="Itqan-community/Munajjam"

echo "🗑️  Deleting ALL issues (open and closed) in $REPO..."
echo ""

# Get all issues (open and closed) and delete them
gh issue list --repo $REPO --state all --limit 500 --json number --jq '.[].number' | while read issue_num; do
    echo "Deleting issue #$issue_num..."
    gh issue delete $issue_num --repo $REPO --yes
done

echo ""
echo "✅ All issues deleted!"
echo ""
echo "View at: https://github.com/$REPO/issues"

