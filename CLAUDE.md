# Package Setup

- Create `.claude/settings.local.json` with the following permissions:
  ```json
  {
    "permissions": {
      "allow": [
        "Bash(julia:*)",
        "Bash(quarto:*)",
        "Bash(grep:*)",
        "Bash(gh api:* --method GET:*)",
        "Bash(gh run view:*)",
        "Bash(gh pr view:*)",
        "Bash(gh issue view:*)",
        "WebSearch"
      ]
    }
  }
  ```

- This package was generated from https://github.com/RallypointOne/JuliaPackageTemplate.jl
- Replace all instances of `JuliaPackageTemplate` to the name set by the user
- Use Julia to create a new UUID
- Enable GitHub Pages deployment from the `gh-pages` branch:
  ```
  gh api repos/{owner}/{repo}/pages -X POST -f source.branch=gh-pages -f source.path=/
  ```
  If Pages is already enabled, update it:
  ```
  gh api repos/{owner}/{repo}/pages -X PUT -f source.branch=gh-pages -f source.path=/
  ```
- Set the repository website URL to the GitHub Pages site:
  ```
  gh repo edit {owner}/{repo} --homepage https://{owner}.github.io/{repo}/
  ```
- Remove "Deployments" and "Packages" from the repo homepage sidebar:
  ```
  gh api repos/{owner}/{repo}/environments/github-pages -X DELETE
  gh api repos/{owner}/{repo} -X PATCH -F "has_deployments=false"
  ```

# Development

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
- Build docs: `quarto render docs`
- `docs/` has its own Project.toml for doc-specific dependencies.
- Each .qmd file in the docs should have `engine: julia` in the YAML frontmatter
- Quarto YAML reference: https://quarto.org/docs/reference/
- Never edit Project.toml or Manifest.toml manually — use Pkg
- Only use top-level .gitignore file
- For Claude's plan mode, always write a "plan_$task.md" in .claude

# Benchmarks

1. Create the `benchmark/` directory with a `Project.toml` and `run.jl`:
   ```
   julia --project=benchmark -e 'using Pkg; Pkg.add(["BenchmarkTools", "JSON3"]); Pkg.develop(path=".")'
   ```
2. Create `benchmark/run.jl` that defines a `BenchmarkGroup` suite, runs it, and writes `benchmark/results.json` (see the template repo for an example)
3. Copy `benchmark/push_results.sh` from the template repo — it pushes `results.json` to the `benchmark-results` orphan branch via a git worktree
4. Run benchmarks locally:
   ```
   julia --project=benchmark benchmark/run.jl
   bash benchmark/push_results.sh
   ```
5. Copy `docs/resources/benchmarks.qmd` from the template repo
6. The Docs workflow automatically includes the benchmarks page when `benchmark-results` branch exists and `docs/resources/benchmarks.qmd` is present — no `_quarto.yml` changes needed

# Docs Sidebar

- `api.qmd` must always be the last item before the "Resources" section in `_quarto.yml`
- `api.qmd` lives in its own `part: "API"` to visually separate it from other doc pages

# Style

- 4-space indentation
- Docstrings on all exports
- Use `### Examples` for inline docs examples
- Segment code sections with: "#" * repeat('-', 80) * "# " * "$section_title" on a single line

# Releases

- First released version should be v0.1.0
- Preflight: tests must pass and git status must be clean
- If current version has no git tag, release it as-is (don't bump)
- If current version is already tagged, bump based on commit log:
  - **Major**: major rewrites (ask user if major bump is ok)
  - **Minor**: new features, exports, or API additions
  - **Patch**: fixes, docs, refactoring, dependency updates (default)
- Commit message: `bump version for new release: {x} to {y}`
- Generate release notes from commits since last tag (group by features, fixes, etc.)
- Important: For major or minor version bumps, release notes must include the word "breaking"
- Update CHANGELOG.md with each release (prepend new entry under `# Unreleased` or version heading)
- Register via:
  ```
  gh api repos/{owner}/{repo}/commits/{sha}/comments -f body='@JuliaRegistrator register

  Release notes:

  <release notes here>'
  ```
