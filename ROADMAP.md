# CS5720 Home Assignment 1 — Phased Roadmap

This is the working plan for the assignment. Each phase says **what** you build,
**why** it exists in the course, the **concepts** you need to be able to explain on
video, the **code approach**, and a **done check**. Work the phases in order; each one
is a single notebook section and a single commit.

Deliverables at the end: a GitHub repo (this folder), a README with your student
info, commented code, saved charts, and a 2–3 minute demo video for Brightspace.

---

## Phase 0 — Environment and repo skeleton

**What:** A reproducible Python environment with TensorFlow + Matplotlib, and a git
repo whose first commit is this scaffold.

**Why:** The grader may run your code. A `requirements.txt` and a one-line run
command in the README means it works on their machine, not just yours. Committing
the skeleton first gives you an honest, incremental history to show in the video.

**Steps**

1. Create and activate a venv (PowerShell):
   ```powershell
   cd C:\Users\jeffr\Downloads\CS5720_HA1_Agnitsch
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If `pip install tensorflow` fails on Python 3.13, either install Python 3.12 and
   build the venv with `py -3.12 -m venv .venv`, or do the whole assignment in Google
   Colab (TensorFlow and TensorBoard are preinstalled there).
2. Verify:
   ```powershell
   python -c "import tensorflow as tf, matplotlib; print(tf.__version__)"
   ```
   Expect a 2.x version. This machine has no NVIDIA GPU, so training runs on CPU.
   That is fine: every model here is small and trains in under a minute.
3. Initialize the repo:
   ```powershell
   git init
   git add .
   git commit -m "chore: scaffold CS5720 HA1 (roadmap, README, requirements)"
   ```
4. Create an empty GitHub repo named `CS5720_HA1_Agnitsch`, add it as `origin`, push.

**Done check:** `python -c "import tensorflow"` works, `git log` shows one commit,
repo is visible on GitHub.

---

## Phase 1 — Tensor manipulation and reshaping

**What:** Build a random (4, 6) tensor, inspect rank/shape, reshape to (2, 3, 4),
transpose to (3, 2, 4), broadcast a (1, 4) tensor onto it, and explain broadcasting.

**Why:** Every layer in a neural network is a tensor operation. Batch dimensions,
image channels, and weight matrices all depend on knowing what shape you have and how
an op changes it. Most beginner bugs in deep learning are shape bugs.

**Concepts to be able to explain**

- **Tensor:** an n-dimensional array. Scalar = rank 0, vector = rank 1, matrix =
  rank 2, and so on.
- **Rank vs shape:** rank is the *number* of axes (`tf.rank(t)`); shape is the
  *length* of each axis (`t.shape` or `tf.shape(t)`). A (4, 6) tensor has rank 2.
- **Reshape:** keeps the elements in the same memory order and just regroups them.
  4×6 = 24 = 2×3×4, so the reshape is legal. Nothing moves.
- **Transpose:** reorders the *axes*, which does move data. `tf.transpose(t, perm=[1, 0, 2])`
  turns (2, 3, 4) into (3, 2, 4) by swapping axes 0 and 1 and leaving axis 2 alone.
- **Broadcasting:** how TensorFlow adds tensors of different shapes without copying.
  Rules, applied from the *rightmost* axis leftward:
  1. If one tensor has fewer axes, pad its shape on the left with 1s.
  2. Two axes are compatible if they are equal or one of them is 1.
  3. An axis of size 1 is "stretched" (virtually repeated) to match the other.

  So (1, 4) against (3, 2, 4): pad to (1, 1, 4), then 1→3, 1→2, 4=4. Result (3, 2, 4).
  The same (1, 4) tensor would **fail** against the original (4, 6) because 4 ≠ 6 on
  the last axis. Say this in the video; it shows you understand the rule rather than
  just running the code.

**Code approach**

```python
tf.random.set_seed(42)                 # reproducible for the video
t = tf.random.uniform((4, 6))
print(tf.rank(t), t.shape)             # rank 2, (4, 6)
r = tf.reshape(t, (2, 3, 4))
p = tf.transpose(r, perm=[1, 0, 2])    # (3, 2, 4)
small = tf.random.uniform((1, 4))
added = p + small                      # broadcasting happens here
```
Print rank and shape after every step. Optionally show `tf.broadcast_to(small, p.shape)`
to make the virtual stretching visible.

**Done check:** printed ranks/shapes read 2 (4,6) → 3 (2,3,4) → 3 (3,2,4) → 3 (3,2,4),
and a markdown cell explains the three broadcasting rules in your own words.

---

## Phase 2 — Loss functions and sensitivity

**What:** Define `y_true` and a few versions of `y_pred`, compute MSE and categorical
cross-entropy for each, print the values, and draw a grouped bar chart.

**Why:** The loss is the number the optimizer minimizes. Choosing the wrong loss for
the task (MSE for classification, for instance) trains slowly or badly. Seeing how each
loss reacts to the *same* small change in predictions is the intuition behind that.

**Concepts to be able to explain**

- **MSE (mean squared error):** mean of (y_true − y_pred)². Natural for regression.
  Bounded when predictions are probabilities, so it is fairly "flat" near wrong answers.
- **Categorical cross-entropy:** −Σ y_true · log(y_pred) over classes, for one-hot
  targets and softmax probabilities. It only looks at the probability assigned to the
  *correct* class, and the log makes confident wrong answers extremely expensive.
- **Why they differ:** moving the correct-class probability from 0.9 to 0.6 changes
  MSE by a little and CCE by a lot. That is why classifiers use cross-entropy: the
  gradient stays large exactly when the model is confidently wrong.

**Code approach**

- `y_true = [[0, 0, 1], [0, 1, 0]]` (one-hot, 2 samples, 3 classes).
- Three prediction sets: *good* (0.9 on the right class), *slightly off* (0.7), *bad*
  (0.3, mass on a wrong class). Each row must sum to 1.
- Use `tf.keras.losses.MeanSquaredError()` and `tf.keras.losses.CategoricalCrossentropy()`.
- Print a small table, then a grouped bar chart (x = prediction set, two bars per group).
  Save it to `images/losses.png` for the README.

**Done check:** CCE grows much faster than MSE as predictions worsen, and the chart
and a two-sentence interpretation are in the notebook.

---

## Phase 3 — Same model, two optimizers (Adam vs SGD)

**What:** Load MNIST, build one small dense network, train it twice with identical
settings except the optimizer, and plot train/val accuracy per epoch for both.

**Why:** The optimizer decides how the loss gradient becomes a weight update. This
phase is a controlled experiment: change one variable, keep everything else fixed, and
compare. That habit is the core of hyperparameter tuning.

**Concepts to be able to explain**

- **MNIST:** 60k train / 10k test images of handwritten digits, 28×28 grayscale,
  10 classes. Pixels are 0–255; divide by 255 so inputs are 0–1 (keeps gradients
  well-scaled).
- **Model:** `Flatten(28×28→784) → Dense(128, relu) → Dense(10, softmax)`.
  Loss: `sparse_categorical_crossentropy` because labels are integers, not one-hot.
- **SGD:** one global learning rate, same step size for every weight. Simple, but
  slow on a plain setup and sensitive to the learning-rate choice.
- **Adam:** keeps a running mean (momentum) and running variance of each weight's
  gradient and scales the step per weight. Usually converges much faster with the
  default lr = 0.001 and no tuning.
- **Expected result:** Adam reaches ~97–98% val accuracy in 1–2 epochs; SGD at
  lr = 0.01 climbs more slowly and is still behind at epoch 5. Be honest in the
  write-up that SGD with a higher lr or momentum narrows the gap; the point is
  *out-of-the-box* behavior.

**Code approach**

- A `build_model()` function so both runs use the exact same architecture.
- `tf.random.set_seed(42)` before each build so initial weights match.
- `model.fit(x_train, y_train, epochs=5, validation_split=0.1)`; keep the returned
  `history` objects.
- One chart with four lines: Adam train, Adam val, SGD train, SGD val. Save to
  `images/adam_vs_sgd.png`.

**Done check:** two history objects, one chart, and a short paragraph stating which
optimizer won and why, citing the numbers from your run.

---

## Phase 4 — TensorBoard logging and analysis

**What:** Train the same MNIST model for 5 epochs with a TensorBoard callback writing
to `logs/fit/<timestamp>`, launch TensorBoard, and answer the three questions.

**Why:** Printing loss to the console does not scale. TensorBoard is the standard
tool for watching training live, comparing runs, and catching overfitting early.
Showing it on screen is the most visual part of your video.

**Concepts to be able to explain**

- **Callback:** a hook Keras calls at the end of each batch/epoch. The TensorBoard
  callback writes scalar summaries (loss, accuracy, for train and val) to disk.
- **Why a timestamped subfolder:** each run gets its own directory so TensorBoard can
  overlay multiple runs instead of overwriting.
- **Overfitting signature:** training loss keeps falling while validation loss flattens
  or starts rising; training accuracy keeps climbing while validation accuracy plateaus.
  The gap between the two curves is the tell.

**Code approach**

```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tb])
```
Launch from the terminal (with the venv active):
```powershell
tensorboard --logdir logs/fit
```
Open http://localhost:6006, go to **Scalars**, and screenshot the accuracy and loss
panels into `images/`. In a notebook you can also use `%load_ext tensorboard` and
`%tensorboard --logdir logs/fit`.

**Answer the three questions from your own curves, not from theory alone:**

1. *Patterns:* training accuracy rises every epoch; validation accuracy rises fast then
   flattens near 97–98%; a small gap opens between them.
2. *Detecting overfitting:* overlay train vs val loss in Scalars; when val loss stops
   decreasing (or rises) while train loss keeps dropping, the model is memorizing.
   Histograms of weights growing large are a secondary signal.
3. *More epochs:* training accuracy approaches 100% and training loss keeps dropping,
   but validation metrics stall and val loss eventually rises. More epochs help until
   the val curve turns, then hurt. Early stopping or regularization is the fix. If you
   have time, actually run 15 epochs and show the turn; that makes the answer concrete.

**Done check:** `logs/fit/` contains at least one run, TensorBoard shows four curves,
screenshots are saved, and the three answers are written in the notebook and README.

---

## Phase 5 — README, comments, video, submission

**What:** Finish the README, pass over every code cell for comments, record the video,
push, and submit on Brightspace.

**Why:** The rubric explicitly weights comments and the README. The video is where
you prove the understanding, not just the output.

**README must contain:** name, student ID, course/section; one-paragraph summary; how
to run (venv + `pip install -r requirements.txt` + open the notebook); a section per
task with the chart image and 2–4 sentences of interpretation; the broadcasting
explanation; the three TensorBoard answers.

**Comment standard:** every cell starts with a one-line purpose comment; every
non-obvious line (reshape, perm, callback, validation_split) gets a short *why*, not a
restatement of *what*.

**Video script (target 2:30)**

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | README top | Who you are, what the four tasks are. |
| 0:15–0:45 | Task 1 cell + output | Rank vs shape, reshape vs transpose, the broadcasting rule and why (1,4) fits (3,2,4). |
| 0:45–1:15 | Task 2 chart | Same prediction change, CCE moves far more than MSE, and why classifiers use CCE. |
| 1:15–1:50 | Task 3 chart | One model, two optimizers, Adam converges faster and why (per-weight adaptive steps). |
| 1:50–2:25 | TensorBoard in browser | Train vs val curves, where overfitting would show, what more epochs do. |
| 2:25–2:40 | GitHub repo page | Commit history, README, done. |

Record with OBS or Windows Game Bar (Win+G). Do a single dry run first.

**Submission checklist**

- [ ] Student info in README
- [ ] All four sections run top-to-bottom with *Restart & Run All*
- [ ] Charts saved in `images/` and embedded in README
- [ ] `logs/fit/` is git-ignored (large, regenerable) but the screenshots are committed
- [ ] Every cell commented
- [ ] Repo pushed; link opens in an incognito window
- [ ] Video 2–3 minutes, uploaded with the link on Brightspace, before the deadline

---

## Suggested commit sequence

```
chore: scaffold CS5720 HA1 (roadmap, README, requirements)
feat(task1): tensor rank, reshape, transpose, broadcasting
feat(task2): MSE vs categorical cross-entropy comparison chart
feat(task3): MNIST Adam vs SGD accuracy comparison
feat(task4): TensorBoard logging and analysis answers
docs: complete README with results, charts, and student info
```
