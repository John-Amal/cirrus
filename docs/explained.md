# What this is, in plain language

No background needed. If you know what rain is and roughly what a computer
model does, that's enough.

## The short version

Weather models that are trained to be *accurate on average* turn out to be
unable to predict heavy rain. Not just bad at it — their maths quietly
assumes rain has a maximum, which it doesn't.

This project shows that, measures how large the error is, and shows what to
change to fix it.

## Start with a guessing game

Imagine I ask you to guess tomorrow's rainfall, and I'll score you by how
far off you are. Not sometimes — every single day, for years, and I add up
all your misses.

You quickly learn the winning strategy: **guess small.** Most days it barely
rains, so small guesses are usually close. The occasional downpour will cost
you, but betting on downpours that don't happen costs you far more often.

A cautious guesser wins that game. A bold one loses.

Now notice what's happened. Your scoring rule has trained a forecaster who is
systematically wrong about exactly the days people care about.

## The same thing happens to computers

A weather model is trained by making guesses and being corrected. The
correction rule — the thing that says *how wrong were you* — is a choice made
by whoever builds it. The standard choice scores the model on average error,
which is that same guessing game.

So the model learns the same lesson: hedge. If it knows a storm is somewhere
in a region but not precisely where, the safest move is to spread a little
rain across the whole region instead of putting a lot of rain in one spot.

It gets rewarded for blurring. You can see it happening — in the figures in
the main README, the real rainfall has sharp, intense spots, and the model's
version is a smooth smear in roughly the right place.

## How bad is it?

There's a standard way to express rare events: the **1-in-20-year storm**.
The amount of rain so heavy you'd expect it about once every twenty years.
It's the sort of number used to decide how big to build a drain, or how much
a flood might cost.

The models trained the usual way get that number about **a third too small**.
And the rarer the event, the worse they do — the 1-in-20 is missed by more
than the 1-in-1.

Worse than the size of the error is its *shape*. Fit a curve to what these
models produce and the curve has a hard ceiling: a rainfall amount they
simply cannot exceed, no matter what. Real rain has no such ceiling. The
model isn't just underestimating, it's describing a different world.

## The fix

Stop asking the model for one number.

Instead of "how much rain?", ask "what are the chances?" — a 60% chance of
any rain, most likely around 2 mm, with a small chance of 30 mm. Then score
it on whether the whole range of possibilities was right, rather than on
whether one number was close.

That removes the reward for hedging. The model can now say *heavy rain,
somewhere around here* without being punished for the uncertainty. When the
same model is trained this way, it reproduces the 1-in-20-year storm
correctly.

That's the finding: **the difference isn't a better model, it's a better
question.** Same neural network, same data, same amount of training — only
the scoring rule changed.

## How the experiment was set up

A fair comparison has to change one thing at a time, so:

1. One model was trained to understand the atmosphere generally — by hiding
   three quarters of the globe and making it fill in the gaps, over 36 years
   of weather. To do that it has to learn how wind, pressure and moisture
   fit together.
2. That understanding was then **frozen**. Six small predictors were attached
   to it, all identical except for the scoring rule used to train them.
3. All six were judged on years of weather the model had never seen —
   2017 to 2022, held back from the start and used once, at the end.

Two deliberately stupid forecasts were included as reference points. One
simply says *the next six hours look like the last six*. The other ignores
today entirely and quotes the historical average for the time of year. Any
real model should beat both — and they all did.

The simple one is more useful than it sounds. It reproduces extreme rainfall
perfectly, because it's just repeating real weather. But it's a poor
forecast. That proves getting extremes right is easy on its own, and only
means something when a model does it *and* forecasts well.

## What it can't do

It predicts **rain only**, **six hours ahead**, and **once** — you can't chain
it forward to get a week. The grid squares are about 600 km across, so it
describes regions, not towns. And it was trained on past weather, so whether
it holds up in a warmer future is a separate question, not yet answered.

It also isn't competing with the national weather service. It's a small model
built to answer one question carefully, not to forecast your weekend.

## Why bother

Because the big operational AI weather models — the ones from Google,
Microsoft and the European forecasting centre — are mostly trained with the
scoring rule that causes this. A forecast system that can't represent a
severe storm is dangerous in a way its average scores will never reveal:
it looks excellent right up until the day it matters.

---

The technical version is in [the README](../README.md), with full results in
[`results/phase3.md`](results/phase3.md) and
[`results/phase4.md`](results/phase4.md).
