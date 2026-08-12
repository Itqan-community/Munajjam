# Contributing to Munajjam

Thank you for your interest in contributing to **Munajjam**! 

Munajjam is an official participant project in the **Code Serves Quran (كود يخدم القرآن)** summer campaign organized by the [Itqan Community](https://community.itqan.dev). We welcome contributions from developers of all skill levels—from open-source beginners making their first Pull Request to experienced backend and AI engineers.

> 🌟 **Code Serves Quran Campaign (August 10 – October 10, 2026)**
> *"You might write a single line of code that benefits thousands of Muslims, accruing continuous reward as long as it is utilized."*
> 
> Join developers worldwide in building open-source Quranic technologies. Participants gain practical experience with GitHub workflows, build their portfolios, connect with the Itqan developer community, win AI tool subscriptions for top contributions, and receive official certificates of participation from the Itqan Community.

---

## 🚀 About Munajjam

**Munajjam (منجّم)** is an open-source Python library and backend API system for automated, high-precision synchronization of Quranic audio recitations with Ayah and word text.

Using AI speech recognition (Whisper / ASR models) and dynamic time warping alignment algorithms, Munajjam automatically calculates exact start and end timestamps for every Ayah and word in long recitations. It eliminates manual timing synchronization, corrects timing drift, and outputs standardized JSON timing data suitable for Quran apps, educational tools, and audio search platforms.

---

## 📖 Table of Contents

- [About Munajjam](#-about-munajjam)
- [Ways to Contribute](#-ways-to-contribute)
- [Contribution Workflow & Guidelines](#-contribution-workflow--guidelines)
  - [1. Finding & Requesting an Issue](#1-finding--requesting-an-issue)
  - [2. Technical Workflow](#2-technical-workflow)
  - [3. Post-Merge Community Sharing](#3-post-merge-community-sharing)
  - [4. Issue Management & Rules](#4-issue-management--rules)
- [Testing Your Changes (Google Colab & Frontend)](#-testing-your-changes-google-colab--frontend)
  - [Why Google Colab?](#why-google-colab)
  - [Step-by-Step Cloud Testing Guide](#step-by-step-cloud-testing-guide)
  - [Mandatory PR Testing Requirement](#mandatory-pr-testing-requirement)
- [Local Development Setup](#-local-development-setup)
  - [Prerequisites](#prerequisites)
  - [Installation & Virtual Environment](#installation--virtual-environment)
  - [Running the API Server Locally](#running-the-api-server-locally)
  - [Pre-commit Hooks & Code Quality](#pre-commit-hooks--code-quality)
  - [Running Tests](#running-tests)
- [Community & License](#-community--license)

---

## 🛠 Ways to Contribute

We welcome contributions across all roles—including Web, Mobile (iOS/Android/React Native), Backend/AI, QA, and DevOps engineers. 

You can contribute to Munajjam in several ways:
- **Developing New Features:** Implementing missing capabilities, API enhancements, or algorithm improvements.
- **Fixing Bugs:** Resolving issues, edge cases, alignment errors, or runtime exceptions.
- **Performance Optimization:** Improving audio processing speed and memory efficiency.
- **Documentation:** Writing, enhancing, or translating technical guides and code documentation.
- **Automated Testing:** Writing unit, integration, or regression tests with `pytest`.
- **Issue Reporting:** Testing recitations, discovering bugs, and submitting clear issue reports.
- **Code Review & Consultation:** Reviewing PRs from fellow contributors or providing technical feedback.

---

## 📋 Contribution Workflow & Guidelines

### 1. Finding & Requesting an Issue

1. **Explore Open Issues:** Browse open tasks in the repository's [Issues](https://github.com/Itqan-community/munajjam/issues) tab.
2. **Check Assignment Status:** Ensure the issue is not marked as `Assigned` and check existing comments to verify no active work is ongoing.
3. **Request Assignment:** Leave a comment on the issue asking the maintainer to assign it to you before starting work.
   - *Important:* Do NOT start coding an issue without being assigned first to prevent duplicate efforts.

### 2. Technical Workflow

1. **Fork & Branch:** Fork the repository and create a new feature branch for your issue:
   ```bash
   git checkout -b feature/issue-123-fix-drift
   ```
2. **Implement & Test:** Make your changes following PEP 8 style guidelines, run local tests, and verify formatting.
3. **Submit a Pull Request (PR):** Push your branch to your fork and submit a PR to the target base branch.
4. **Code Review:** Follow up on feedback from project maintainers until your PR is approved and merged.

### 3. Post-Merge Community Sharing

Once your PR is merged, share your contribution story on the **Itqan Community** platform!

Your community post should include:
- Link to the merged PR and original Issue.
- Your experience (especially if this was your first open-source contribution).
- Technical overview of how you solved the issue and learnings gained.
- Advice for fellow contributors.

### 4. Issue Management & Rules

- **Prevent Issue "Hoarding":** Only request assignment for an issue if you plan to start working on it immediately. Do not "reserve" issues for future weeks.
- **AI-Assisted Code Policy:** Review, understand, and test all code before submitting. Do not submit unverified AI-generated code.
- **Proposing New Features:** Open an Issue first to propose new ideas or report bugs, and wait for maintainer feedback before coding.

---

## 🧪 Testing Your Changes (Google Colab & Frontend)

> 📌 **Official Main Guide:** For complete details, screenshots, and updates on cloud testing, visit the [Itqan Community Colab Guide](https://community.itqan.dev/d/648).

### Why Google Colab?

Munajjam uses AI speech recognition models (Whisper / ASR) that require GPU acceleration for fast processing. To allow everyone to test code changes without needing local GPU hardware, we provide a cloud testing workflow using Google Colab connected to a live frontend application.

### Step-by-Step Cloud Testing Guide

#### Step 1: Open the Frontend Interface
- **Munajjam Frontend App:** [https://alinice1998.github.io/munajjamfrontend/](https://alinice1998.github.io/munajjamfrontend/)
- **Frontend Repository:** [https://github.com/alinice1998/munajjamfrontend](https://github.com/alinice1998/munajjamfrontend)

#### Step 2: Retrieve and Update Colab Code Snippets
1. On the frontend page, choose a Riwayah and click **Colab Code Snippets (أكواد التشغيل COLAB)**.
2. Copy the two generated code snippets.
3. **Important:** In Snippet #1, replace the official repository URL with **your fork's repository URL**:
   ```bash
   # Update the URL to point to your modified fork:
   git clone https://github.com/YOUR_GITHUB_USERNAME/munajjam-backend.git
   ```

#### Step 3: Set Up Google Colab Runtime
1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Navigate to **Runtime > Change runtime type**.
3. Select **T4 GPU** under Hardware accelerator and click **Save**.

#### Step 4: Run Code Snippet #1 (Setup Environment)
1. Paste modified **Snippet #1** into a Code cell and click **Run** (takes 2–4 minutes).
2. *Note:* If Colab shows a prompt asking to restart the session/runtime, click **Cancel** (do NOT restart).

#### Step 5: Run Code Snippet #2 (Launch Colab Server)
1. Paste **Snippet #2** into a second Code cell and click **Run**.
2. Copy the public server URL displayed in the output within seconds.

#### Step 6: Test Synchronization in Frontend
1. Return to the [Munajjam Frontend](https://alinice1998.github.io/munajjamfrontend/).
2. Paste the Colab Server URL into the **Colab Server URL** field.
3. Select a Surah, upload your test recitation audio file (MP3/WAV), and click **Start Synchronization (بدء المزامنة)**.

#### Step 7: Verify Alignment Accuracy
- During playback, observe word-by-word real-time text highlighting.
- Click individual words or Ayahs to jump audio playback and confirm timing precision.

### Mandatory PR Testing Requirement

Before submitting a Pull Request, **you MUST test your code changes on at least:**
1. **One short Surah** (e.g., Surah Al-Fatiha or Surah Al-Ikhlas).
2. **One long Surah** (e.g., Surah Al-Baqarah, Yasin, or Maryam).

Verify that the issue is completely resolved and timing alignment remains precise without drift.

---

## 🛠 Local Development Setup

### Prerequisites

- **Python:** 3.10 or higher
- **FFmpeg:** Installed and available in system PATH
- **Git**

### Installation & Virtual Environment

```bash
# 1. Clone your fork
git clone https://github.com/YOUR_GITHUB_USERNAME/munajjam-backend.git
cd munajjam-backend

# 2. Create virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Install package in editable mode with dev dependencies
cd munajjam
pip install -e ".[dev]"
cd ..

# 4. Install FastAPI server dependencies
pip install fastapi uvicorn python-multipart
```

### Running the API Server Locally

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Access interactive documentation at `http://localhost:8000/docs`.

### Pre-commit Hooks & Code Quality

```bash
pip install pre-commit
pre-commit install
```

Automated checks run on every commit:
- **Ruff:** Linting & import sorting
- **Ruff Format:** Code formatting
- **Mypy:** Static type checking
- **General Checks:** Trailing whitespace, EOF fixer, YAML check

### Running Tests

```bash
pytest
pytest tests/unit/test_arabic.py -v
```

---

## 🤝 Community & License

- **Official Colab Testing Guide:** [Itqan Community Discussion #648](https://community.itqan.dev/d/648)
- **Itqan Community:** [community.itqan.dev](https://community.itqan.dev)
- **Discord:** [Join Itqan Discord](https://discord.gg/24CskUbuuB)
- **GitHub Issues:** [Munajjam Issues](https://github.com/Itqan-community/munajjam/issues)
- **Contact:** [connect@itqan.dev](mailto:connect@itqan.dev)

### License

Contributions are licensed under the [MIT License](LICENSE).
