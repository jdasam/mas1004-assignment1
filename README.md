# Assignment 1: Train an image classifier on your own images

MAS1004 Data & AI, 2026 Fall

Due: Friday 16 October, 22:00, on Cyber Campus (LMS).

## What you do

You choose 3 to 5 categories of image, collect images of them, and train a
classifier that tells them apart. The model is ResNet18, a network that was
already trained on ImageNet, 1.2 million photographs of 1,000 kinds of thing.
You replace its last layer so that it answers with your categories, and train
it further on your images. Then you put it on a web page that anyone can open,
and find out what it actually looks at when it decides.

Two things matter most:

- Categories that come from your own interest: something you care about or
  know well, such as a hobby, your major, or things you see every day and can
  tell apart yourself. Please avoid categories chosen only because they are
  easy to separate, such as cats and cars.
- Finding out which visual features your model uses, by changing images on
  purpose and watching its answers (Problem 7). You do this part yourself.

3 to 5 is the number of categories in one classifier. One model that tells
espresso cups, wine glasses and paper coffee cups apart has 3 categories. You
do not run three separate projects on different topics.

## What you hand in, on Cyber Campus (LMS)

1. `links.txt`, a text file with the address of your web demo and the address
   of your code repository, in this form:

   ```
   demo: https://<your name>.github.io/<your repository>/
   code: https://github.com/<your name>/<your repository>
   ```

   The model runs inside the web page, so the demo works without a server, and
   nobody's photos are sent anywhere.
2. The short report as a PDF: one page, written by you, from the template in
   `short_report/`.
3. The long report as a PDF: written with your agent, from the template in
   `long_report/`.

Both templates are LaTeX files that you compile on Overleaf
(https://www.overleaf.com), with a free account. The sections "The short
report" and "The long report" at the end of this file say how.

## How you work

`PLAN.md` is the plan for the whole assignment, step by step, with a check for
each step. Read it, then give it to your coding agent one step at a time.
`AGENTS.md` tells your agent how to run things in this repository. Most agents
read it on their own.

You write the bodies of these functions. Each has a docstring that says what
goes in and what comes out, and a test that checks it.

| file | functions | problem |
|---|---|---|
| `src/data.py` | `prepare_image`, `load_folder`, `split_train_test` | 2 |
| `src/train.py` | `build_model`, `train` | 3 |
| `src/evaluate.py` | `predict_logits`, `accuracy`, `confusion_matrix` | 3 |
| `src/evaluate.py` | `worst_examples` | 5 |

Everything else is given: the image downloader, the cleaning tool, the
training script, the export script, the web page and the checker.

You are responsible for everything you hand in, including what your agent
wrote.

- Every number in your reports has to come from output that you saw printed
  by code you ran. Agents sometimes report numbers they never measured.
- When the agent says it fixed something, run the test yourself and see it
  pass.
- You make the guesses and choose the image changes in Problem 7, and you
  write the short report. The agent makes only the image changes you ask for.

## Getting your own copy

This repository is a template, so you do not fork it and you do not work in it.

1. Make an account at https://github.com if you do not have one.
2. Open https://github.com/jdasam/mas1004-assignment1
3. Press "Use this template", then "Create a new repository".
4. Give it a name and keep it public. Public is what makes GitHub Pages free.
5. `git clone` your new repository onto your own computer.

Everything you do from now on happens in your copy, and you hand in its
address. Commit and push as you go.

### Updating the given files

When the given files change, take them from the template into your copy. In
your repository folder:

```
git remote add template https://github.com/jdasam/mas1004-assignment1.git
git fetch template
git checkout template/main -- README.md PLAN.md AGENTS.md CLAUDE.md .gitignore .python-version pyproject.toml uv.lock requirements.txt pytest.ini assignment1_colab.ipynb src/collect.py src/clean.py src/check.py src/run.py src/export_web.py docs/index.html docs/app.js tests short_report long_report/mas1004.sty
git commit -m "Update the given files"
```

The first line is needed only the first time. This replaces only the files
named on the third line, none of the ones you write, and it does not touch
your images.

If your copy has no `long_report` folder yet, take it once as well. Do not do
this again after you have started filling in the long report, because it
replaces `long_report/long_report.tex`:

```
git checkout template/main -- long_report
git commit -m "Add the long report template"
```

If your `src/data.py` has no `prepare_image` in it, your copy is from before
the starter code changed to ResNet18. Make a new copy from the template and
move your `data/` folder into it.

## Setting up

On your own computer, use uv. It installs the right version of Python and of
every package into a `.venv` folder inside your repository, the same on
Windows, macOS and Linux. Install it once. On macOS, in Terminal:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, in PowerShell:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close the terminal, and quit and reopen VS Code. Then, in your repository
folder:

```
uv run pytest tests/test_export.py
```

The first time takes a few minutes. Six tests should pass. They cover code
that was given to you, so if they fail, ask for help before you go on.

Intel Macs and Windows laptops with an ARM processor (Snapdragon) cannot run
PyTorch, and Macs with Apple Silicon need macOS 14 or later. On those
computers, use Colab.

### Google Colab

Colab gives you a GPU for free, and training ResNet18 is much faster on a GPU.
Open the notebook directly in Colab with this link:
https://colab.research.google.com/github/jdasam/mas1004-assignment1/blob/main/assignment1_colab.ipynb

Use Colab only to run things: downloading, training, cleaning and checking.
Write your code on your own computer with your agent, and push it. The
notebook takes your code from your repository on GitHub, so push before you
run it, and do not edit code in Colab: the notebook replaces it with what is on
GitHub every time it updates. When the training is done, you download the
model files from Colab and publish the page from your own computer
(Problem 6).

---

## Problem 1. Choose your categories and collect the images

Choose 3 to 5 categories, as described at the top. The downloader searches the
web for each category and saves the images, and you work on a copy of them.

Two things to think about when you choose. In Problem 5 you test the model on
images from a source that cannot overlap with your downloads, such as your own
photos, so choose things you can find such images of. And the model can only
use what is visible in the picture: an expensive wine glass and a cheap one
that look the same cannot be told apart by any model. Because you start from a
network that has already seen 1.2 million photos, categories that differ in
small details, such as similar breeds of dog, are worth trying.

Write this down:
- Why these categories? Why do you care?
- What do you think the model will actually use to tell them apart: shape,
  colour, texture, background, or something else? Write it before you train.
  Problem 7 tests it.
- How many images did you get for each? Show the counts.

## Problem 2. Turn your folder into numbers

Write `prepare_image`, `load_folder` and `split_train_test`. `prepare_image`
prepares a photo exactly the way every ImageNet photo was prepared when
ResNet18 was trained: the shorter side resized to 256, the 224 by 224 square
in the middle cut out, and each colour channel scaled. The network can only
read your photos if they are prepared the same way.

No image may be in both the training set and the test set. If one is, your
test accuracy measures what the model remembers, not what it learned.

Write this down:
- One of your images next to what `prepare_image` made of it. What was cut
  off? Find an image where the crop cut off part of the thing you care about.

## Problem 3. Train it and report the first result

Write `build_model`, `train`, `predict_logits`, `accuracy` and
`confusion_matrix`. Then train at least four times, before you clean anything:
starting from the ImageNet weights, starting from random numbers, training
only the new last layer, and one setting of your own. Fill in this table.

| tag | start | trained | epochs | lr | trainable parameters | train accuracy | test accuracy |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Write this down:
- The loss curve. Did the loss go down? Did it keep going down, or did it
  flatten out?
- Your train accuracy is higher than your test accuracy. By how much? What does
  that gap mean?
- The confusion matrix. Which two categories does it mix up most? Does that
  surprise you?
- The same images, the same code, and the same number of epochs, starting from
  ImageNet and starting from random numbers. How far apart are the two test
  accuracies? Why do you think that is?
- Training only the last layer changes about 0.01% of the parameters. How
  close does it get to training all of them?

## Problem 4. Clean your data and train again

Every downloaded image got its label from a search engine. You typed "wine
glass", and whatever came back is now called `wine_glass`: drawings, product
catalogues, charts, photos of a bar, the same picture four times with
different watermarks. The model learns all of them as true examples, and your
test set contains them too.

Only you know what you meant by each category, so you write a rule first and
then decide each image by it. "I removed pictures where the object was not the
main thing in the frame" is a rule. "I removed the ones that looked wrong" is
not. Look at every image yourself. The cleaning tool also lists the images that
look least like the rest of their class, and pairs of near copies, but a
suspect is not automatically junk: decide it by your rule.

Cleaning changes your test set too, so compare the models before and after
cleaning on your new images from Problem 5, which do not change.

Write this down:
- Your rule, in one or two sentences, as you wrote it before you started.
- Three kinds of junk you found, with one picture of each.
- The table of how many images you removed from each class, and why.
- How many near copies it found, and what you did with them.
- Of the first 20 suspects in each class, how many did you remove? Did the
  suspects include junk you had missed when you looked yourself? Did you find
  junk that it did not list?
- The test accuracy before and after, and the accuracy on your new images
  before and after. If a number went down, say so and think about why. That
  happens, and an honest explanation is worth more than a good number.

## Problem 5. Test it on images from a new source

Your test set came from the same search as your training set, so it shares its
habits: the same kind of product photo, the same white backgrounds. Now test
the model on images from somewhere else.

Collect at least 5 images per category from a source that you can be sure is
not in your downloads: photographs you take yourself, photographs a friend
took, frames from a video you recorded, or photos from your phone's gallery.
Another search engine or another search phrase does not work, because the same
photos are copied onto every shopping site. Collect them early, because
Problem 4 uses them too.

Save them as JPEG or PNG. An iPhone saves photos as HEIC unless you set
Settings, Camera, Formats to Most Compatible, and the code here cannot read
HEIC. Your agent can convert ones you already took.

Your accuracy here will probably be worse than your test accuracy from
Problem 3. That is not a mistake you made. Explaining it is the assignment.

Write this down:
- Where your new images came from, and why you are sure none of them are among
  your downloads. The output of the overlap check.
- The accuracy on your downloaded test set and the accuracy on your new images,
  side by side.
- Five mistakes it was confident about, with the pictures. For each one, what
  in that image do you think pushed it the wrong way?
- What is different about your new images? Background, lighting, angle,
  distance, what else is in the frame?

## Problem 6. Put the demo on the web

Export your cleaned model to the page and open it on your own computer first.
If you trained on Colab, copy the three model files from Colab first (the last
section of the notebook). The badge at the top of the page has to be green. A
red badge means the page prepares images differently from your
`prepare_image`, and every answer on it is wrong.

Then commit and push, and turn on GitHub Pages: your repository on the GitHub
website, Settings, Pages, Source "Deploy from a branch", Branch `main`, Folder
`/docs`, Save. Your demo appears at
`https://<your name>.github.io/<your repository>/` after a minute or two.
Open it on your phone to check that it works away from your computer.

Write this down:
- Your GitHub Pages address.
- One sentence on what happens when you show it something that is none of your
  categories.

## Problem 7. Find out what your model looks at

This is the most important problem in the assignment.

In Problem 1 you wrote down what you thought the model would use to tell your
categories apart. Now you have a trained model and you have seen its mistakes.
In this problem you change images on purpose and see how its answers change,
to find out what it actually uses.

Do this part yourself, not by handing it to your agent. You look at the images
and the mistakes, you make the guess, and you choose the change that tests it.
The agent only makes the changes you ask for.

1. Write down a guess that is specific enough for an image change to prove it
   wrong. "It uses the shape" is too vague. "It tells a wine glass from an
   espresso cup by the long stem" can be tested.
2. Choose the change that would show whether the guess is right, and the images
   to change. Some kinds of change, depending on the guess:
   - one part of the object: cover that part with a plain grey box, or crop it
     out
   - the background: put the same object on a different background
   - colour: turn the image grayscale, or shift its colours
   - fine texture: blur the image
3. Ask your agent for exactly that change, on images you name. For example:
   "Make a copy of each image in data/new_images/wine_glass with the bottom
   third covered by a grey rectangle, and save the copies in
   data/changed/wine_glass_no_stem."
4. Upload the original and the changed images to your demo page and compare
   its answers. The page also shows the square the model actually sees. Check
   that the part you changed is inside it.
5. Use several images for each change, not one. A single image can change its
   answer for reasons that have nothing to do with your guess.

A result that goes against your guess tells you as much as one that supports
it. Write it down, and make a new guess from it.

Write this down:
- Each guess, and what in your images or your model's mistakes made you think
  of it.
- For each change: the images you used, what you changed, why that change
  tests the guess, and the model's answers before and after, with the original
  and the changed images side by side.
- The exact requests you gave your agent in this problem.
- What you now think the model uses to tell your categories apart, and how sure
  you are.

---

## The short report

One page. Not one and a bit. One.

Write it yourself. No language model, at any stage, including for tidying it up
afterwards. If Korean is your first language, write it in Korean. This is the
one piece of work in this course where neither your English nor your polish
counts for anything, and what you actually think counts for everything.

Four questions:

1. What does your model tell apart, and why did you pick that?
2. The worst mistake it makes. Which image, what did it answer, and what do
   you think made it answer that?
3. One thing your coding agent got wrong. What did it claim, what was actually
   true, and what made you look?
4. If you started again tomorrow, what would you do differently?

On the third question: everybody's agent gets something wrong. It writes code
that runs and does the wrong thing, or it reports an accuracy it never
measured, or it quietly changes something you told it not to. "Nothing went
wrong" will be read as "I did not check".

Write it on Overleaf, from the template. Log in to Overleaf, then open one of
these links. Each makes a new project of your own with the template in it:

- [English template](https://www.overleaf.com/docs?engine=xelatex&main_document=short_report_en.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/short_report_en.tex&snip_name%5B%5D=short_report_en.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/mas1004.sty&snip_name%5B%5D=mas1004.sty)
- [Korean template](https://www.overleaf.com/docs?engine=xelatex&main_document=short_report_ko.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/short_report_ko.tex&snip_name%5B%5D=short_report_ko.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/mas1004.sty&snip_name%5B%5D=mas1004.sty)

If a link does not work, compress the `short_report` folder into a zip, and
on Overleaf choose New Project, Upload Project. Either way, the compiler has to
be XeLaTeX (Menu, Compiler). Do not use Overleaf's AI tools on this report. If
it runs past one page, a red note appears at the end of the PDF.

## The long report

As long as you like, written with your agent. It holds everything the "Write
this down" boxes ask for, in order, with the pictures and the complete output
of the checker. Every number in it has to be one you saw printed by code you
ran.

The template is `long_report/long_report.tex`. Your agent fills it in, in your
repository: it copies the pictures into `long_report/figures/` and saves the
output of each command into `long_report/outputs/`, from where the report
prints it exactly as it was. Red text in the PDF is what is still missing.

To make the PDF, compress the `long_report` folder into a zip, and on Overleaf
choose New Project, Upload Project. Set the compiler to XeLaTeX (Menu,
Compiler) and press Recompile.

## How this is graded

There are no points attached to the problems. Four things are looked at:

- Problem 7: whether your guesses come from your own images and mistakes,
  whether each image change actually tests its guess, and what you concluded
  from the results
- Whether the demo works, at the address you gave, on someone else's computer
- Whether the numbers in your long report are the ones `check.py` actually
  prints
- The short report

A model that scores 95% with no explanation is worth less than one that scores
55% whose owner can tell you exactly which pictures it fails on and why.

## When you are stuck

- Run `uv run python src/check.py`. It works out the answers for itself from
  the files on disk, so when it disagrees with your own code, one of the two
  is wrong.
- If a test fails, give your agent the whole test output, not your summary of
  it.
- Ask in class. Both Tuesday and Thursday have time for this.
