# Contributing to fix a specific issue

General note: This software is developed to work on linux and released under the MIT license. All third party software you use in your contributions must work
on Debian Linux, 64 bit, Python 3.7+. You may not use any packages released under a copyleft license
like GPL or AGPL. Allowed licenses include LGPL, Apache, MIT, MPL, BSD. If you want to
use a package released under another license, ask the maintainer (Elias Hohl).

All upcoming tasks for this project can be found at
https://git.ehtec.co/research/pie-chart-ocr/-/issues

Developers of this project should only take care of issues assigned to them by the
Maintainer (Elias Hohl).

To contribute to an issue, you need to create a new branch (for example:
`0-example-issue`) from master to do your changes.

If you don't have `git` installed yet, install it via your package manager. On Debian or
Ubuntu:
```commandline
sudo apt install git
```

To contribute for the first time, execute these commands:
```commandline
git clone https://git.ehtec.co/research/pie-chart-ocr
git checkout -b 0-example-issue
```

Replace `0-example-issue` with your own branch name. If you have already cloned the
directory before, execute these command instead:
```commandline
git pull
git checkout -b 0-example-issue
```

Now you can start making your changes.

When you have made some changes to the code, you can make a commit (a message
associated with your code changes):

```commandline
git add .
git commit -m "example commit message"
```

The use of an IDE like PyCharm simplifies this process.

As soon as you are ready or if you have a question, go forward with the next steps
(you should create a merge request before starting any discussion of your code, because
it is easier for other developers to track what you are doing if you have created a
draft merge request).

To make sure your code is compliant with the coding guidelines, run `flake8` locally:

```commandline
python3 -m flake8 .
```

You can install `flake8` with this command:

```commandline
python3 -m pip install flake8
```

You might want to back up your working directory locally before going forward with the
next steps.

You are ready now to push your changes:

```commandline
git push -u origin 0-example-issue
```

Go to the Gitlab issues page and select your issue. In the upper right corner, click
the button to create a merge request. Mark the merge request as draft if it is not yet
ready. Then press "compare branches and continue". The Asignee of the merge request
should be the maintainer (Elias Hohl). Otherwise, no notification to review your work
will be sent.

If Gitlab shows you that the branch is not up to date with master and needs to be
rebased, press the "Rebase" button. If you encounter any merge conflicts,
resolve them. You can continue pushing commits that will automatically be
integrated into the same merge request.

In case the "Rebase" button is greyed out and you are notified that you need to rebase
locally, execute the following steps:

```commandline
git checkout master
git pull origin master
git checkout 0-example-issue
git rebase origin/master
```

Fix the rebase conflicts. The command will pause everytime a conflict occurs. Open the
affected file in a text editor, fix the conflict, and then execute

```commandline
git rebase --continue
```

until the rebase succeeds. Then continue by pushing your changes:

```commandline
git push -u origin 0-example-issue
```

Gitlab is running `flake8` automatically (see
https://git.ehtec.co/research/pie-chart-ocr/-/pipelines). A merge request
that does not comply with the coding guidelines will be blocked automatically. In this
case, you need to continue pushing commits until the problems are resolved.

Gitlab is also utilizing `nose2` and `unittest` to automatically execute unit tests.
The tests are executed when a merge request is created / open for the branch or when
creating a tag / release (only possible for maintainers). All unit tests should be
placed in `tests/unit/test_<name_of_file_to_test.py>`. Future integration tests will
be placed in `tests/integration/`. Data used for the tests (for example test pie
charts) should be placed in `test_data/`.

If you want to learn more about automated testing, read this:
https://realpython.com/python-testing

A project-specific example:
https://git.ehtec.co/research/pie-chart-ocr/-/blob/main/tests/unit/test_basefunctions.py

To try out the unit tests:
```commandline
python3 -m nose2 --start-dir tests/ --with-coverage
```

The request will be merged by the maintainer (Elias Hohl). Do not push anything else
before the merge is completed, as new commits will automatically be added to the merge
request.

After the merge has completed, do not forget to execute `git pull` before proceeding
with new changes.
