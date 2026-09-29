# The plan

This is the plan I would write before starting this assignment. Give it to your
coding agent one step at a time, not all at once. An agent handed twelve steps
will do a bad job of all twelve.

Read it first. Part of what this course is teaching is how to write one of
these yourself, and you cannot learn that by pasting it without looking.

Each step says what to do, and how you know it worked. The second part matters
more. A step is not done because the agent says it is done. It is done when the
check passes.

---

## Step 0. Make sure the room is empty before you move in

Run `pytest tests/test_export.py`. Six tests should pass. They cover code you
were given, so if they fail, nothing you write afterwards will work either.

Check: six passed.

## Step 1. Choose your categories

Not a coding step. Write down 3 to 5 categories and one sentence each on why
you chose them. Keep the sentences. They go in your report.

Check: you can say out loud what visible difference the model is supposed to
find between them.

## Step 2. Download the images

```
python src/collect.py --classes "your,categories,here" --n 150
cp -r data/raw data/clean
```

Check: `data/clean/` has one folder per category and each has at least 50 files
in it. If one category came back nearly empty, the search phrase is wrong.
Change the phrase, not the category.

## Step 3. Look at what you downloaded

Open the folders and scroll through them. Do not skip this. You are looking for
what is in there that should not be, and you will need it in Problem 4.

Check: you can name three kinds of junk that came back.

## Step 4. Write `load_folder`

Give the agent the docstring of `load_folder` in `src/data.py` and ask it to
write the body. Then run:

```
pytest tests/test_data.py -k "not split"
```

The tests will fail in specific ways. Give the agent the failure text as it is,
not your summary of it.

Check: every test in that file except the split ones is green.

## Step 5. Write `split_train_test`

Same again, then `pytest tests/test_data.py`.

Before you move on, read the test called
`test_split_has_no_image_on_both_sides` and make sure you understand why it
exists. If you do not understand why an image being in both halves is a
problem, ask the agent to explain it, and do not accept an answer you cannot
repeat.

Check: all of `tests/test_data.py` is green.

## Step 6. Write `build_model` and `train`

```
pytest tests/test_train.py
```

The last test trains on three easy blobs and expects over 90%. If your training
loop has a bug, that is where it shows up. Common ones: forgetting
`optimiser.step()`, forgetting `optimiser.zero_grad()`, training on the test
set, or measuring accuracy as a percentage instead of a share.

Check: all of `tests/test_train.py` is green.

## Step 7. Write `predict_logits`, `accuracy` and `confusion_matrix`

```
pytest tests/test_evaluate.py -k "not worst"
```

Check: green.

## Step 8. Run the whole thing for the first time

```
python src/run.py
```

Look at `results/run_curves.png` and `results/run_confusion.png`.

Check: it finished, it printed a train accuracy and a test accuracy, and the
loss curve goes down. If the two accuracies are identical, look at your split
again.

## Step 9. Run it four more times with different settings

```
python src/run.py --size 16 --tag small
python src/run.py --gray --tag gray
python src/run.py --hidden 256 64 --tag deep
python src/run.py --epochs 80 --tag long
```

Write each printed row into the table in your report as you go. Do not wait
until the end and try to remember.

Check: four more rows in the table, and you can say which setting mattered most.

## Step 10. Clean the data and run again

Delete the junk from `data/clean`, then:

```
python src/run.py --tag clean
```

Check: you wrote down your deletion rule and how many you removed, before you
compare the numbers.

## Step 11. Take your own photographs

At least five per category, into `data/my_photos/<category>/`. Use the same
folder names.

Check: `python src/check.py` finds them and reports an accuracy.

## Step 12. Write `worst_examples` and look at the mistakes

```
pytest tests/test_evaluate.py
python src/run.py --tag clean
```

Check: `results/clean_worst.png` shows ten pictures with what the model said
and what they really are.

## Step 13. Open the demo on your own machine

```
python -m http.server -d docs 8000
```

Open http://localhost:8000.

Check: the badge at the top is green. If it is red, stop and fix it. A red badge
means the page is not preparing images the way you prepared them for training,
so everything it says is wrong. Read what the badge says, and check that the
`--size` and `--gray` of your last run match what `docs/model.json` says.

## Step 14. Publish it

```
git add -A
git commit -m "trained model and demo"
git push
```

Then on the GitHub website: your repository, Settings, Pages, Source "Deploy
from a branch", Branch `main`, Folder `/docs`, Save.

Your agent can do the git part for you. If it goes in circles over `gh auth
login` for more than ten minutes, stop it and use the website. Getting the
command line tool authenticated is not what this assignment is about.

Check: you opened the address on your phone, away from your own wifi, and it
worked. If the page is blank, look at your repository on the GitHub website and
see whether `docs/model.json` and `docs/weights.bin` are actually there.

## Step 15. Write the long report

Go back through the "Write this down" boxes in README.md in order, with your
agent. You should already have every number and every picture you need, from
the checks above.

Check: every number in it is one you saw printed by code you ran.

## Step 16. Write the short report

Close the agent. One page, by yourself, in Korean if that is your first
language. The four questions are in README.md.

This is the last step for a reason. You cannot answer question 2 or 4 until you
have seen your model fail, and you cannot answer question 3 unless you were
paying attention the whole way through.

Check: it fits on one page, and you wrote all of it.
