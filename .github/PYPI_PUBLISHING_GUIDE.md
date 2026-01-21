# PyPI Publishing Guide

This guide explains how to set up automatic PyPI publishing for the LRDBenchmark package.

## Overview

The repository includes GitHub Actions workflows that automatically publish packages to PyPI when:
- A GitHub release is published
- A workflow is manually triggered (with optional test-only mode)

## Setup Instructions

### 1. Create PyPI API Tokens

#### For Production PyPI:
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Create a token with scope "Entire account" or "Project: lrdbenchmark"
4. Copy the token (it starts with `pypi-`)

#### For TestPyPI (recommended for testing):
1. Go to https://test.pypi.org/manage/account/token/
2. Create an account if you don't have one
3. Create an API token
4. Copy the token

### 2. Add Secrets to GitHub

1. Go to your GitHub repository: https://github.com/dave2k77/lrdbenchmark
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:
   - **Name**: `PYPI_API_TOKEN`
     **Value**: Your PyPI API token (starts with `pypi-`)
   - **Name**: `TEST_PYPI_API_TOKEN`
     **Value**: Your TestPyPI API token

### 3. Publishing Workflows

#### Workflow Files:
- **`.github/workflows/publish-to-pypi.yml`**: Main publishing workflow
- **`.github/workflows/test-pypi-publish.yml`**: Test publishing workflow

#### Triggering a Release:

**Option 1: Create a GitHub Release (Recommended)**
1. Ensure `pyproject.toml` has the correct version number
2. Create a git tag: `git tag v2.3.0`
3. Push the tag: `git push origin v2.3.0`
4. Go to GitHub → **Releases** → **Draft a new release**
5. Select the tag you just created
6. Fill in release notes
7. Click **Publish release**
8. The workflow will automatically build and publish to PyPI

**Option 2: Manual Workflow Dispatch**
1. Go to **Actions** tab in GitHub
2. Select **Publish to PyPI** workflow
3. Click **Run workflow**
4. Optionally specify a version or enable test-only mode
5. Click **Run workflow**

**Option 3: Test Publishing First**
1. Use the **Test PyPI Publishing** workflow
2. Push a version tag: `git tag v2.3.0 && git push origin v2.3.0`
3. Or manually trigger it from the Actions tab
4. Test install: `pip install --index-url https://test.pypi.org/simple/ lrdbenchmark==2.3.0`

## Version Management

### Updating Version

The version is read from `pyproject.toml`:
```toml
[project]
version = "2.3.0"
```

**Important**: Always update the version in `pyproject.toml` before creating a release tag.

### Version Tag Format

Use semantic versioning with a `v` prefix:
- `v2.3.0` (major.minor.patch)
- `v2.3.1` (patch release)
- `v2.4.0` (minor release)
- `v3.0.0` (major release)

## Testing Before Publishing

### Local Testing:
```bash
# Build the package locally
python -m build

# Check the package
twine check dist/*

# Upload to TestPyPI (requires TestPyPI account)
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ lrdbenchmark==2.3.0
```

### Using TestPyPI Workflow:
The test workflow will publish to TestPyPI automatically when you push a version tag, allowing you to verify the package before publishing to production PyPI.

## Troubleshooting

### Workflow Fails with "403 Forbidden"
- Check that your `PYPI_API_TOKEN` secret is correctly set
- Verify the token hasn't expired
- Ensure the token has the correct scope

### Package Already Exists
- The workflow uses `skip-existing: true`, so it won't fail if the version already exists
- If you need to overwrite, you'll need to delete the version from PyPI first (if allowed)

### Version Mismatch Warning
- The workflow will warn if the tag version doesn't match `pyproject.toml`
- It will use the version from `pyproject.toml` to ensure consistency

### Build Errors
- Check that all dependencies in `pyproject.toml` are correct
- Verify that the package structure is correct
- Check the workflow logs for specific error messages

## Best Practices

1. **Always test on TestPyPI first** before publishing to production
2. **Update version in pyproject.toml** before creating a release tag
3. **Write comprehensive release notes** when creating GitHub releases
4. **Tag releases with semantic versioning** (v2.3.0 format)
5. **Monitor workflow runs** to ensure successful publication

## Security Notes

- Never commit API tokens to the repository
- Use GitHub Secrets for all sensitive information
- Rotate tokens periodically
- Use project-scoped tokens when possible (not account-wide)

## Additional Resources

- [PyPI Documentation](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-cicd/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging Guide](https://packaging.python.org/)

