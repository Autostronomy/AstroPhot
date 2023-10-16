# Contributing to AstroPhot

Thank you for taking time to help make AstroPhot better for everyone! Your contributions are appreciated and once merged with the main branch you will be officially recognized as a contributor to the project!

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR. The workflow for contributing to AstroPhot is quite standard, nonetheless we include details here for clarity.

## Setup

To begin working on the development of AstroPhot, you will need to install the developer version of the code. 
Instead of simply using `pip install astrophot` you will instead fork and clone the main branch.
Follow the instructions [at this link](https://docs.github.com/en/get-started/quickstart/fork-a-repo) to fork AstroPhot, once this is done you will have a version of the code that you can modify freely.
Next clone the forked repo to your machine, if you are unfamiliar with this process, the instructions are also in the fork-a-repo link.
Once you have cloned to a suitable location, cd into the root folder and run `pip install -e .` this tells pip to install AstroPhot in an editable format.
Any changes you make to the AstroPhot code base will be reflected the next time you use `import astrophot as ap` or however you interact with the code.

## Main Workflow

### Creating an Issue

Issues are the main method used to track the addition of new features, improvement of current functionality, and debugging problems. 
If you spot something that should be improved/added, especially if you know how to do it, please create an Issue using the `Issues` tab above.
To do this, first check if there are any similar issues already in place.
If so, consider adding a comment to that issue to expand its scope, or you can assign the issue to yourself if you are ready to tackle it.
If no issue already exists, then make a new one; make sure to use a descriptive title and include lots of keywords even if it makes the grammar a bit funny.
Then add a detailed description of what needs to be done; a detailed description is one that someone other than yourself could take and work from to add the new feature.
Make sure to tag your issue with any tags that are relevant for that issue.
If you are ready to take on the Issue, then assign yourself to it.
Please don't assign other people to an issue unless they have explicitly asked you to.

### Solving an Issue

To begin solving an Issue make a new branch with a concise name that reflects the Issue at hand.
Once you have created the new branch, go to the `Issues` tab and add a comment to the issue specifying the branch name that you have created.
You may then begin making changes in the new branch to solve the Issue.
Please regularly commit your code and include helpful messages so that someone reading through the git log will have a good sense of what has been done.
Please also try to format your code in a similar manner to the rest of the code. For docstrings we use google format, for code formatting we use black.

You are not alone in solving the Issue! 
Feel free to contact other contributors (especially Connor Stone) and ask for advice.
You may also wish to add comments to the Issue as you progress since this is the most visible way that others can see your progress and comment on any ideas/modifications.

Finally, when you are nearing completion of the issue, please add unit tests which reach the newly added/modified code.
These should live in the `tests` directory, and you can run all the unit tests by simply calling `pytest` from the command prompt while in that directory.
Also, make sure to update any relevant documentation if you have changed/added a feature that is public facing.

### Commit the Solution

Once you believe you have solved the issue, please make sure that all the unit tests still run, including any you have added.
Then make sure to merge in the main branch to your branch.
This will add in any new features that have been pushed since you started working.
Please resolve any conflicts that arrise and ensure that the unit tests still run.
Commit this final version as the solution to the Issue.

### Pull Request

Mark the branch as ready for review and create a pull request.
Make sure to link the PR and the Issue you are solving.
Please make sure to allow maintainer edits so updates can be added before merging.
Discussion on the PR will then determine if it is ready to merge, following that the merge will be completed!
Since you already merged the main branch before submitting the PR, there should be no conflicts to resolve.
You are done!
