#!/bin/bash

# Close all open issues in the repository
# Usage: ./scripts/close_all_issues.sh

REPO="Itqan-community/Munajjam"

echo "🗑️  Closing all issues in $REPO..."
echo ""

# Get all open issue numbers and close them
gh issue list --repo $REPO --state open --limit 100 --json number --jq '.[].number' | while read issue_num; do
    echo "Closing issue #$issue_num..."
    gh issue close $issue_num --repo $REPO
done

echo ""
echo "✅ All issues closed!"
echo ""
echo "View at: https://github.com/$REPO/issues"

