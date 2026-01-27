# GitHub Repository Setup Instructions

## Repository has been initialized locally ✅

Your git repository is ready! Follow these steps to connect it to GitHub:

## Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in:
   - **Repository name**: `CSE156_PA1_WI26` (or your preferred name)
   - **Description**: "CSE156 Assignment 1: Neural Networks for Text Classification"
   - **Visibility**: Private (recommended for assignments) or Public
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
cd /Users/james/Projects/CSE156_PA1_WI26

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/CSE156_PA1_WI26.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/CSE156_PA1_WI26.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify

1. Go to your repository on GitHub
2. You should see all your files including:
   - README.md
   - Python source files
   - Data files
   - Configuration files

## Future Updates

When you make changes, commit and push:

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Important Notes

- **Data files**: The data files (train.txt, dev.txt, embeddings) are included. If they're too large (>100MB), GitHub may warn you. Consider using [Git LFS](https://git-lfs.github.com/) for large files.
- **Private repo**: For assignments, it's recommended to keep the repository private
- **.gitignore**: Already configured to exclude:
  - Virtual environment (`.venv/`)
  - Python cache files (`__pycache__/`)
  - Generated plots (`.png` files)
  - OS files (`.DS_Store`)

## Troubleshooting

### If you get authentication errors:
- Use a [Personal Access Token](https://github.com/settings/tokens) instead of password
- Or set up [SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

### If repository already exists:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/CSE156_PA1_WI26.git
```

### To check your remote:
```bash
git remote -v
```
