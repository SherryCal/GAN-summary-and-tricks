# for Pre
The basic idea behind adversarial machine learning is that it extend machine learning from one player which has one cost function representing their interests.\

instead, adversarial machine learning  dealing with more than one player with more than one cost function, on the left we show what the loss function looks like in a traditional machine learning algorithm, we have some kinds of cost function that takes player parameters and describes how well that player performs. \
Higher cost means the worse performance so that maybe something like the negative log likelihood assigned to the labels on a training data set.\
For example, what is the negative log probability the model will assign the correct labels to all of the different images in an object recognition data set this is the way we train things like classifiers and many different kinds of generative models and even some kinds of reinforcement learning
we show what happens when we start to have more than one player and more than one cost. \
So a player might be something like a machine learning model or it might also be a person or a program that is trying to interfere with the operation of a machine learning model you will see a lot of concrete examples. But what we can think about easily is the example of spam detection. \
we have machine learning model that wants to recognize spam and we have spammers who want to get their spam through the system\
we can model this at the language of game theory and draw a value function where we are looking for a point called Nash equilibrium that is simultaneously a minimum of the defending players cost and a maximum of the attackers attacking player cost. \
So, this is for example a point where the spammers can not get anymore spam through the system unless they were somehow able to change the spam detector and the spam detector is not able to get anymore accuracy unless it was somehow or other able to change the spam generation algorithms used by the spammers.

# for Pap
