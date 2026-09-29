# Assignment 1: Train an image classifier on your own images

MAS1004 Data & AI, 2026 Fall

Due: Friday 16 October, 22:00, on LMS.

## What you will hand in

A web page, on the internet, at an address you can send to anyone. It looks at
a picture and says which of your categories it thinks it is. The picture can
come from your webcam, from a file you upload, or from something you draw with
the mouse. The model runs inside the page itself, so it keeps working after you
close your laptop, and nobody's photos are sent anywhere.

The model is one you trained, on images you collected, of categories you chose.

You will also hand in two reports. One page that you write yourself, and a long
one that you write with your agent. More on both at the bottom.

## What you are given and what you write

You are given the parts where there is nothing to learn and a lot to get stuck
on: the image downloader, the export script, the web page, the checker, and a
test for each function you have to write.

You write four things:

| file | functions | problem |
|---|---|---|
| `src/data.py` | `load_folder`, `split_train_test` | 2 |
| `src/train.py` | `build_model`, `train` | 3 |
| `src/evaluate.py` | `predict_logits`, `accuracy`, `confusion_matrix` | 3 |
| `src/evaluate.py` | `worst_examples` | 5 |

Each one has a docstring saying exactly what goes in and what comes out, and a
test that checks it. Read the docstring, ask your agent to write the body, run
the test, repeat until it is green.

`PLAN.md` is a plan for doing all of this, written out step by step. Give it to
your agent one step at a time. It is a real plan of the kind you should learn
to write yourself, so read it before you use it.

## Getting your own copy

This repository is a template, so you do not fork it and you do not work in it.

1. Make an account at https://github.com if you do not have one.
2. Open https://github.com/jdasam/mas1004-assignment1
3. Press "Use this template", then "Create a new repository".
4. Give it a name and keep it public. Public is what makes GitHub Pages free.
5. `git clone` your new repository onto your own machine.

Everything you do from now on happens in your copy, and you hand in its
address. Commit and push as you go. A commit you did not push has not been
handed in.

## Setting up

```
pip install -r requirements.txt
```

If `pip` is not available, or you would rather not touch your system Python,
`uv run --with-requirements requirements.txt python src/run.py` works too.

Check that it is alive:

```
pytest tests/test_export.py
```

Those tests cover code that was given to you, so they should pass before you
write a single line. If they do not, fix that first and ask for help if you
need it. Everything else will fail until you write it.

---

## Problem 1. Choose your categories and collect the images

Choose 3 to 5 categories. They have to be things you care about for some
reason, and you have to say what that reason is.

```
python src/collect.py --classes "espresso cup,wine glass,paper coffee cup" --n 150
```

This downloads images into `data/raw/`. Then copy `data/raw` to `data/clean`
and work on the copy, so you always have the original to go back to.

Two warnings about your choice. Categories that are told apart by colour and
overall shape will work far better than categories that are told apart by small
details, because the model you are going to train sees a 32 by 32 picture and
nothing else. And categories that are too close together, such as three breeds
of white dog, will not work at all this time. Choose something you would still
find interesting if the answer turned out to be 60% right.

Write this down:
- Why these categories? Why do you care?
- What do you think the model will actually use to tell them apart?
- How many images did you get for each? Show the counts.

## Problem 2. Turn your folder into numbers

Write `load_folder` and `split_train_test` in `src/data.py`.

```
pytest tests/test_data.py
```

Your images are all different sizes and shapes, some are broken, and one of
them is a text file that ended up with a .jpg name. That is normal. The tests
put you through the same things.

Pay attention to the test called
`test_split_has_no_image_on_both_sides`. If the same picture is in the
training set and the test set, your test accuracy is not a measurement, it is a
memory. This is the single most common mistake in this assignment and it always
makes your numbers look better than they are.

Write this down:
- What did you decide about image size and colour, and why?

## Problem 3. Train it and report the first result

Write `build_model` and `train` in `src/train.py`, and `predict_logits`,
`accuracy` and `confusion_matrix` in `src/evaluate.py`.

```
pytest tests/test_train.py tests/test_evaluate.py
python src/run.py
```

`src/run.py` trains, measures, draws three pictures into `results/`, and
exports the web files. Run it at least four times with different settings and
fill in this table. Every run prints the row for you.

| image size | colour | hidden layers | parameters | train accuracy | test accuracy |
|---|---|---|---|---|---|
| | | | | | |

Useful things to change: `--size 16` and `--size 64`, `--gray`,
`--hidden 32` and `--hidden 256 64`, `--epochs`, `--lr`.

Write this down:
- The loss curve from `results/run_curves.png`. Did the loss go down? Did it
  keep going down, or did it flatten out?
- Your train accuracy is higher than your test accuracy. By how much? What does
  that gap mean?
- The confusion matrix from `results/run_confusion.png`. Which two categories
  does it mix up most? Does that surprise you?

## Problem 4. Clean your data and train again

Look at your images. Actually look at them, all of them. Downloaded images
contain drawings, logos, collages, screenshots, pictures of the wrong thing,
and the same picture four times. Delete what does not belong in `data/clean`.

Write down your rule before you start deleting, and follow it. "I removed
pictures where the object was not the main thing in the frame" is a rule.
"I removed the ones that looked wrong" is not.

Then run `python src/run.py --tag clean` and compare.

Write this down:
- Your deletion rule, in one or two sentences.
- How many images you removed from each category.
- The accuracy before and after. If it went down, say so and think about why.
  That happens, and an honest explanation is worth more than a good number.

## Problem 5. Test it on photographs you took yourself

Everything so far used images from the internet. Now take your own.

Take at least 5 photographs per category yourself and put them in
`data/my_photos/<category>/`, using exactly the category names from your
training folders. Write `worst_examples` in `src/evaluate.py`, then run:

```
python src/check.py
```

It measures your model on your own photographs and prints a confusion matrix
and the mistakes it was most confident about.

Your accuracy here will almost certainly be much worse than your test accuracy
from Problem 3. That is the point of this problem, and it is not a mistake you
made. Explaining it is the assignment.

Write this down:
- The accuracy on internet images and the accuracy on your own photographs,
  side by side.
- Five mistakes it was confident about, with the pictures. For each one, what
  in that photograph do you think pushed it the wrong way?
- What is different about your photographs? Background, lighting, angle,
  distance, what else is in the frame?

## Problem 6. Put the demo on the web

`src/run.py` already wrote everything the page needs into `docs/`. Look at it
first on your own machine:

```
python -m http.server -d docs 8000
```

and open http://localhost:8000. Opening `index.html` by double clicking will
not work, and neither will the webcam, because browsers only allow cameras on
`https://` pages and on `localhost`.

Check the badge at the top of the page. If it is red, the page is preparing
images differently from the way you prepared them for training, and the demo is
lying to you. Fix that before you publish.

Then commit and push, and turn on GitHub Pages: your repository, Settings,
Pages, then Source "Deploy from a branch", Branch `main`, Folder `/docs`, Save.
Your demo appears at `https://<your name>.github.io/<your repository>/` after a
minute or two. The first time you look it is often a 404. Wait and reload.

The folder is called `docs` for exactly this reason. GitHub Pages publishes a
folder with that name and no other configuration.

Check that `docs/model.json` and `docs/weights.bin` really are in your
repository on the GitHub website. If you only see `index.html` and `app.js`,
you trained a model but never committed it, and your published page will be
blank.

Write this down:
- Your GitHub Pages address.
- One sentence on what happens when you show it something that is none of your
  categories.

---

## What to submit on LMS

1. The address of your web demo.
2. The address of your code repository.
3. The short report.
4. The long report.
5. `data/my_photos/` as a zip. Not the downloaded images, only your own
   photographs.

## The short report

One page. Not one and a bit. One.

Write it yourself. No language model, at any stage, including for tidying it up
afterwards. If Korean is your first language, write it in Korean. This is the
one piece of work in this course where neither your English nor your polish
counts for anything, and what you actually think counts for everything.

Four questions:

1. What does your model tell apart, and why did you pick that?
2. The worst mistake it makes. Which photograph, what did it answer, and what
   do you think made it answer that?
3. One thing your coding agent got wrong. What did it claim, what was actually
   true, and what made you look?
4. If you started again tomorrow, what would you do differently?

On the third question: everybody's agent gets something wrong. It writes code
that runs and does the wrong thing, or it reports an accuracy it never
measured, or it quietly changes something you told it not to. "Nothing went
wrong" will be read as "I did not check".

## The long report

As long as you like. Write it with your agent. That is what it is for, and it
is the right tool for this job.

Be aware of who reads it. It will be read by a program, and by you while you
are writing the short report. So write it for those two readers: complete,
ordered, every number traceable. Do not write an introduction, and do not
explain what a neural network is.

It holds:

- Everything the "Write this down" boxes asked for, in the order they appear
- Your experiment table, at least four rows
- The pictures from `results/` for every run you refer to
- The complete output of `python src/check.py`
- Your cleaning rule and how many images you removed from each category
- The accuracy on internet images and the accuracy on your own photographs,
  next to each other

## How this is graded

There are no points attached to the problems. Three things are looked at:

- Whether the demo works, at the address you gave, on someone else's computer
- Whether the numbers in your long report are the ones `check.py` actually
  prints
- The short report

A model that scores 95% with no explanation is worth less than one that scores
55% whose owner can tell you exactly which pictures it fails on and why.

## Using AI coding agents

Use them for the code and for the long report. That is what this course is
about. You are responsible for what you submit: you have to be able to explain
what you were trying to do, how you got your result, and what it means, even if
you cannot explain every line.

Three rules follow from that. Never write a number that you did not see printed
by code you ran. When the agent says it fixed something, run the test yourself
before you believe it. And write the short report with your own hands, because
it is the one place where the answer has to be yours.

## When you are stuck

- Run `python src/check.py`. It works out the answers for itself from the files
  on disk, so when it disagrees with your own code, one of the two is wrong.
- Read the error at the bottom of the traceback, not the top.
- If a test fails, paste the whole test output to your agent, not your summary
  of it.
- Ask in class. Both Tuesday and Thursday have time for this.
