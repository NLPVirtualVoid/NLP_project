# NLP_project

This repository is for collaboration on a UCL NLP project lookng at code generation by team VirtualVoid.

## Collaboration on GitHub

### Setting up the repository and a virtual environment:
First, clone the repository:

```bash
git clone https://github.com/NLPVirtualVoid/NLP_project
```
Then set up a virtual environment by opening the Anaconda prompt, changing your directory to your local project directory, and finally
running the following comnand from the Anaconda prompt:
```bash
conda env create -f NLP_project.yml
```

### Git workflow

TBD but ideally we would aim to use a Feature Branch Workflow (see https://info201.github.io/git-collaboration.html).
Failing this (i.e. if people find it too complex) we can fall back to a forking workflow. 

Feature Branch Workflow:

N.B. That the branch called ``master`` in that document is called ``main`` for our project

Standard workflow should work like this:

1. Ensure your copy of main is up to date with the remote and create a feature branch off this for your upcoming changes:

```bash
git checkout main    # switches your local branch to main
git fetch origin     # fetches the state of the remote (central) main branch 
git reset --hard origin/main   # hard reset - forces your local main to match the central repo
git checkout -b feature-<yourname>-dev   # creates a new local branch for development - branched off local copy of main
```

2. Do some work / make some changes to exising files/code or add new files in the repo etc

For example maybe I edited this file : README.md

3. Update your local dev branch:
```bash
git add <thefile>   # i.e. <> = README.md . Can add multiple files in a single commit
git commit -m "Some msg - e.g I updated README.md"
```

4. Push changes to GitHub - note this creates a remote feature branch that is tracking your local one.

```bash
git push -u origin  feature-<yourname>-dev
```
Now you can make further changes as needed and update the remote with a simple ``git push``.

5. Ideally one would (manually) set up a pull request (PR) on GitHub at this point. You can request that a member of the team reviews your code prior to it being merged.
   Anyone on the team can test and inspect your changes locally using ``git pull origin feature-<yourname>-dev``.

6. Once there is agreement that the code will be merged to the master codebase use the following steps:
   
```bash
git checkout main  # switch to master branch locally
git pull origin main  # download any changes that may have occurred in the meantime from the central repo

git merge feature-<yourname>-dev  # merge the feature into the master branch
# fix any merge conflicts - hopefully we don't have any of these ! 

git push origin main  # upload the updated code to master
```

7. Once you are satisified that the feature has been merged successfully we should tidy up:
 ```bash
git branch -d feature-<yourname>-dev # delete the dev branch
git push origin :feature-<yourname>-dev # update the remote too

It might seem wasteful to delete the local/remote feature branches if one is only going to recreate them
for the next set of changes but actually this is very helpful for those not superfamiliar with git
because it keeps everything clean 
