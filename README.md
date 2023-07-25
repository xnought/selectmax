# selectmax

Inspired this one dudes tweet on why softmax is giving weirdness in attention

$$\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

In summary, the fact that you need to choose between discrete entities into a probability forces you to weigh stuff high even if it isn't pertinent.

In the case where the query and key shouldn't weigh any values, what do you do? Apparently it was found that certain tokens have extreme spikes (like space tokens) which mess things up.

The dude thinks a +1 fixes a lot of this in softmax.

https://www.evanmiller.org/attention-is-off-by-one.html

## What else

Are there any other formulations that don't use softmax at all?
