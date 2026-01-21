# Repository Rename Guide: LRDBenchmark → lrdbenchmark

This guide will help you rename your GitHub repository from `LRDBenchmark` to `lrdbenchmark`.

## Steps to Rename on GitHub

### 1. Rename the Repository on GitHub

1. Go to your repository: https://github.com/dave2k77/LRDBenchmark
2. Click on **Settings** (top right of the repository page)
3. Scroll down to the **Repository name** section
4. Change the name from `LRDBenchmark` to `lrdbenchmark`
5. Click **Rename**

**Note**: GitHub will automatically:
- Update the repository URL
- Redirect old URLs to the new name
- Update webhooks and services
- Preserve all issues, pull requests, and releases

### 2. Update Your Local Git Remote

After renaming on GitHub, update your local repository's remote URL:

```bash
# Update the remote URL
git remote set-url origin https://github.com/dave2k77/lrdbenchmark.git

# Verify the change
git remote -v
```

### 3. Verify Everything Works

```bash
# Test fetching from the new URL
git fetch origin

# Test pushing (use --dry-run first to be safe)
git push --dry-run origin master
```

### 4. Update Any External Services

If you have any external services configured (CI/CD, webhooks, etc.), they may need to be updated:

- **ReadTheDocs**: Update the repository URL in your ReadTheDocs project settings
- **GitHub Actions**: Should automatically work, but verify workflows
- **PyPI**: URLs in package metadata will be updated automatically when you push
- **Any webhooks**: Update if they reference the repository URL

## What Has Been Updated

All repository URLs in the codebase have been updated from:
- `https://github.com/dave2k77/LRDBenchmark` → `https://github.com/dave2k77/lrdbenchmark`

**Files Updated:**
- `pyproject.toml` - Package metadata and URLs
- `setup.py` - Setup configuration URLs
- `config/pyproject.toml` - Config package URLs
- `README.md` - Installation and support links
- `RELEASE_NOTES_v2.3.0.md` - Changelog links
- `.github/PYPI_PUBLISHING_GUIDE.md` - Publishing guide
- `docs/` - All documentation files
- `notebooks/README.md` - Notebook documentation

## After Renaming

1. **Commit and push the changes**:
   ```bash
   git add -A
   git commit -m "Update repository URLs from LRDBenchmark to lrdbenchmark"
   git push origin master
   ```

2. **Verify the repository is accessible**:
   - Visit: https://github.com/dave2k77/lrdbenchmark
   - Ensure all links work correctly

3. **Test package installation**:
   ```bash
   pip install git+https://github.com/dave2k77/lrdbenchmark.git
   ```

## Important Notes

- **Old URLs will redirect**: GitHub automatically redirects old URLs to new ones
- **Bookmarks may break**: Update any bookmarks you have
- **Clones may need updating**: Anyone who has cloned the repo should update their remotes
- **Package metadata**: The package name `lrdbenchmark` remains unchanged (already correct)

## Troubleshooting

### If you get "repository not found" errors:
- Verify the repository name is exactly `lrdbenchmark` (lowercase)
- Check that you have the correct permissions
- Ensure the rename was successful on GitHub

### If local git commands fail:
- Verify the remote URL: `git remote -v`
- Update it manually: `git remote set-url origin https://github.com/dave2k77/lrdbenchmark.git`

### If external services break:
- Check service logs for repository URL errors
- Update service configurations with the new repository URL

## Next Steps

After renaming, you may want to:
1. Update any documentation that references the old repository name
2. Announce the rename to your users/contributors
3. Update any badges or links in external documentation
4. Verify all CI/CD pipelines still work

