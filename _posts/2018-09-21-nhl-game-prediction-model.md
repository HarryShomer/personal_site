---
layout: post
title: "Predicting Individual NHL Games"
date: 2018-09-21
---


The topic of predicting individual hockey games is one that many find interesting and others 
find very useful. It is also a very hard. Josh Weissbock has [suggested](https://ruor.uottawa.ca/bitstream/10393/31553/3/Weissbock_Joshua_2014_thesis.pdf) 
that a model can theoretically, at best, predict 62% of NHL games correctly. The difficulty in correctly
predicting the outcomes of games is important to keep in mind when evaluating mine and other's
models. 


This topic is also one that contains a fair amount of previous scholarship. Emmanuel Perry has 
[written about his model](https://www.corsicahockey.com/corsica-predictions-explained) that
drives the predictions on his site [Corsica](http://corsica.hockey/). Micah Blake McCurdy has also
[developed](http://hockeyviz.com/txt/edgar) a succession of different models over the past few seasons. 
Lastly, MoneyPuck has [written](http://moneypuck.com/about.htm) as his model as well.
I recommend everyone to take the and read the previous work done of the subject before 
continuing. Many of the ideas here I derived from these works and some will be referenced
in this piece as well.

In order to tackle this problem I decided to create two different models using the data
from 2007-2017. One would focus on the broad team statistics for both teams. The second would focus 
just on the players who  are slated to play for both teams in that given game. The outputs of both 
will then be combined into a final model. So we make two separate sub-models and our final
model will be some combination of the two. 

Team Model
----------

The following features were used in the model for each team:

Even-Strength
* Unblocked Shots For Per 60 Minutes
* Unblocked Shots Against Per 60 Minutes
* Expected Shooting Percentage on Unblocked shots For 
* Expected Shooting Percentage on Unblocked shots Against 
* Shooting percentage / Expected Shooting Percentage on Unblocked shots For 

Power Play
* Unblocked Shots For Per 60 Minutes
* Expected Shooting Percentage on Unblocked shots For
* Shooting percentage / Expected Shooting Percentage on Unblocked shots For

Penalty Kill
* Unblocked Shots Against Per 60 Minutes
* Expected Shooting Percentage on Unblocked shots Against

All Situations
* Penalties Taken Per 60 Minutes
* Penalties Drawn Per 60 Minutes

Misc. 
* Probability of home team winning given Elo Ratings
* Days of Rest for Team
* If the Starting Goalie played in back to back (played in a game the previous day)
* Fsv% - xFsv% - Goalie Marcels (only Even-Strength)

One thing you may notice is that some stats are per 60 minutes. This is a way of standardizing
these numbers across different teams. Each team plays a different amount of time at each
strength so some teams have more opportunities than others (which would happen on a count
or per game basis). So per 60 allows us to properly compare the numbers among two different
teams.

Next, you may see that each strength has different numbers included for each. Concerning 
Even-Strength, I tried to model in a way that made sense to me. On the offensive side, we 
want to see: How many shots a team generates, what's the average expected probability of those shots 
with the probability computed by a [expected goal model](https://github.com/HarryShomer/xG-Model),
and how much better or worse their sh% was than expected. For defense, we also care about
how many shots a team gives up and what the average expected probability of those shots.
But the last metric, how much better/worse than expected, will be handled by including the 
goalie stats later. So it's not included in that form.

For the power play and penalty kill we only examine the stats that are most pertinent to 
the team in question in that situation. On the power play, how a team does defensively 
is a distant second to how well they generate offense (and vice-versa for the penalty kill).
So for the power play we include the three offensive stats listed in the last paragraph and
for the penalty kill just the two defensive one's.  

Some of you may be wondering how we account for goalie quality. What's Fsv% - xFsv%?
And what does marcels means? Let's answer this one at a time.

Fsv% stands for Fenwick Save Percentage which is another way of saying Unblocked Shot Save
Percentage. So instead of just looking at shots on goal we look at all unblocked shots. This 
is done as there is reason to believe that [goalies can effect](http://fooledbygrittiness.blogspot.com/2017/02/a-few-stuff-about-miss.html)
the rate at which shots miss the net. Next, xFsv% just means expected Fsv%. With the expected
probability gotten from my previously mentioned expected goal model. The logic here is that
we want to see which goalies are doing better/worse than what we expect. 

Marcels, is a simple [projection system created](https://www.baseball-reference.com/about/marcels.shtml) 
by Sabermetrician Tom Tango. The mechanics of how it works is beyond the scope of this post
but it's a very simple way of projecting how a player will do (and by extension giving a
best guess at how good a player currently is). Marcels works by weighting past seasons
(traditionally 3 is used but I used 4 because goalie data is rather random), regressing 
towards the mean, and applying a simple aging curve.<sup>[1](#footnote1)</sup>


Team Elo Ratings
================

You may have noticed the feature "Probability of home team winning given Elo Ratings" in
the Misc. category. I think it is worth the time to talk about this in more depth. 
Elo is a popular [chess measure](https://en.wikipedia.org/wiki/Elo_rating_system) which have
been used in many other settings. It has been used in hockey by both [Alan Ryder](http://hockeyanalytics.com/2016/07/elo-ratings-for-the-nhl/)
and recently by [Cole Anderson](http://crowdscoutsports.com/team_elo.php).  

Elo ratings work by starting off each team at a score of 1500. They then gain/lose points
by winning or losing games. The amount they win/lose depends on the ratings of both teams 
and a constant K. The value of this constant depends on competition we are looking at. 

We can derive the probability of one team winning based on their elo ratings with the 
formula: 

```python
def get_home_prob(game, team_elo):
    """
    Get the probability that the home team will win for a given game
    
    Home % of Winning = 1 / (1 + 10^(dr/400)) ; dr = Away_Elo - Home_Elo
    
    :param game: Dict with home and away team data for that game
    :param team_elo: Dict of Elo Ratings
    
    :return probability of home team winning
    """
    home_advantage = 33.5
    dr = team_elo[game['Opponent']] - (team_elo[game['Team']] + home_advantage)

    return 1 / (1 + 10 ** (dr/400))
```

You can see there is also an extra term in there to account for the advantage home teams have.
The number 33.5 can be found by using the equation referenced above and the fact that we know 
the historical home win% from 2007 is 54.8%. We can solve for the difference in ratings 
this translates to - which is 33.5. So the home team, on average, tends to look 33.5 points
better than they actually are. 

But, as mentioned still need K factor and a way to update these ratings after every game. 
Based on [Cole Anderson's work](http://crowdscoutsports.com/team_elo.php) we use the following 
equation:

```python
def update_elo(game, team_elo):
    """
    Update the elo ratings for both teams after a game
    
    :param game: Dict with home and away team data for that game
    :param team_elo: Dict of Elo Ratings
    
    :return Dict with updated elo ratings
    """
    # k is the constant for how much the ratings should change from this game
    # GD = Goal Differential
    # Subtracted by 2 if the game went to a shootout
    k_rating = 4 + (4 * game['GD']) - (game['if_shootout'] * 2)

    # New Rating = Old Rating + k * (actual - expected)
    elo_change = k_rating * (game['if_home_win'] - game['home_prob'])
    team_elo[game['home_team']] += elo_change
    team_elo[game['away_team']] -= elo_change

    return team_elo
```

So the K factor is a function of the goal differential and if the game went to a shootout.
The higher the goal differential, the more sure we are the one team is better or worse 
than the other. And, if the game goes to a shootout we subtract by 2 since it signifies that
the teams may be more evenly matched. 
 
Lastly, before each season I regressed the elo ratings of each team 50% of the way towards
the mean. This is done to account for roster and player changes.   

I originally meant to include the elo derived probability as a separate from the 
player and team models. But, after some testing I found that including it as part of the
team model performed better. I think this makes sense because at the end of the day elo
ratings are essentially just another way of valuing team performance. 


Weighting Team Stats
====================

I still think there is one thing that is unclear. For the few stats in the Misc. category,
it's easy to see the data used. Probability given the elo ratings uses the current ratings,
days of rest checks the schedule, goalie b2b checks the schedule and if the goalie played,
and goalie marcels uses the numbers from the past few years. But, what about all the other 
stats? For example, what data is used for 'Penalties Taken Per 60 Minutes'? Is it just 
that season, the past two seasons, the past 25 games? 

What I chose to do what as follows (this applies to all stats under the Even-Strength, Power Play,
Penalty Kill, and All Situations categories): For a given season, we calculate the stats for every 
game for every team. Then for a game n in that season we get all the games before that given
game (from 1 to n-1) for that season. So for a game played on December 5th between NYR and MTL, 
we get the stats for all the previous games for each team that season before December 5th. 
The question now is how do we combine them. Do we just take the mean for each stat and call it a day? 

Well, I thought about it and decided to weight them exponentially by the equation e^-.05x, 
where x is the number of games between the game we are weighting and the game we are currently
trying to predict. This logic here is similar to how MoneyPuck handles it, thought his weighting 
system is different. Also, for those interested, I arrived at this value of -.05 through some
guesswork and trial and error. Below are some examples of the weights:

```python
>>> import math
	
    # Get the weights for games 0 - 81
    # 0 here means when the game being weighed was the previous game (so there are
    # 0 games in between that game and the current game we are predicting)
>>> season_weights = [round(math.e ** (-.05 * x), 2) for x in range(0, 82)]
>>> season_weights[0]
1.0
>>> season_weights[10]
.61
>>> season_weights[30]
.22
>>> season_weights[50]
.08
    # The weight here is how much the first game of the season matters when trying to 
    # predict the last game of the season (82 - 1 = 81). As you can see, for the 82nd game 
    # how a team did in the first game that season doesn't really matter. 
>>> season_weights[81]
.02
```

There's still one issue though. What about the first game of the season? We have no prior
games there? Also, in general, for games early in the season we don't have a lot of data to go one.
So how do we handle this?

I thought about it and decided to include previous season stats when a team has played less
than 25 games that season. The number 25 is gotten from the fact that a lot of team stats
peak in predictivity at around the 25 game mark (check out the team section [here](http://fooledbygrittiness.blogspot.com/2018/03/evaluating-my-shooter-xg-model.html)).
The equation we use to weight how much the previous stats matter is e^-.175x, where x is 
the games played so far that season. Some examples:

```python
>>> import math
    # x represents how many games were played in that year
>>> prev_season_weights = [round(math.e ** (-.175 * x), 2) for x in range(0, 24)]
    # Since we played no games so far, the previous season counts as 100%
>>> prev_season_weights[0]
1.0
>>> prev_season_weights[5]
.42
>>> prev_season_weights[12]
.12
>>> prev_season_weights[23]
.02
```

As you can see (and just tell from the equation itself) this decays much quicker than our
game equation. This is by design as we don't want last season's numbers to matter too much. 
All we need it for is a little boost in the beginning. 


Player Model
-----------

This model contains the following features for each team:

* Game Score Marcels for Forwards 1-12
* Game Score Marcels for Defensemen 1-6
* Goalie Marcels for Starter & Backup
* If the Starting Goalie played in back to back (played in a game the previous day)

To represent a single number value for each skater I decided to use game score developed by 
[Dom Luszczyszyn](https://hockey-graphs.com/2016/07/13/measuring-single-game-productivity-an-introduction-to-game-score/). 
Game Score takes a bunch of different stats, weights them, and combines them into one single
number. The higher the game score the better. Many will point out that this may not be the
best way of determining the value of a single player, but it has the advantage of being both
easy to understand and to calculate so I chose it here (Note: Dom has also posted game 
probabilities in the past using a model which I'm pretty sure uses game score. So it may 
be similar to what I have here). 

So for each game we calculate the game score marcels for each forward and defensemen for 
both teams only using data from prior to that game.<sup>[2](#footnote2)</sup> The question though is how do we order the skaters in our model. For each team our 
model takes 20 player inputs (12 forwards, 6 defensemen, 2 goalies) so we need a way of 
organizing them into some order. To deal with this I chose to follow a similar method to 
Emmanuel Perry in his previously mentioned model. He chose to rank them in decreasing
order of time on ice played in the past 5 games by position. Instead of just using the last 
5 games I chose to create a time on ice projection using a marcels based model.<sup>[3](#footnote3)</sup> 
I then ranked them in terms of decreasing projected time on ice by forward, defensemen, and the
goalies (with there being 12 forwards, 6 defensemen, and 2 goalies). So the forward with 
the 7th highest projected time on ice for his team is slotted in as the 7th of 20 players, the 3rd 
highest defensemen is the 15th (12+3), and the backup goalie is the 20th. 

An issue we run into this is when there isn't 12 forwards or 6 defensemen dressed for that 
game. Every now and then teams dress 11 forwards and 7 defensemen or 13 forwards and 5 defensemen
(also, crucially, the NHL game rosters used to determine which players dressed for that game
makes mistakes so we see some weird stuff here and there). This isn't a common occurence
so I chose to deal with it by simply shuffling the player to the other position. So, if 
11 forwards were dressed the defensemen with the lowest projected time on ice is placed
as the 12th forward. Not the cleanest methodology, but I'm sure it's fine. 


Fitting the Sub-Models
--------------------------------

So far we have listed the features used for both models, so now is the time to see how we
are going to use those features to create a model. Before we fit anything though we need to
get the data in the way we want it.

For both models, we standardized all the relevant features for that model (below is just
how it is done for the teams). 

```python
def get_team_features(df):
    """
    Get the features for the team model

    :param df: Team df of all features and labels

    :return: features
    """
    continuous_vars = ['FA60_even_Opponent', 'FA60_even_Team',
                       'FA60_pk_Opponent', 'FA60_pk_Team',
                       'FF60_even_Opponent', 'FF60_even_Team',
                       'FF60_pp_Opponent', 'FF60_pp_Team',
                       'GF60/xGF60_even_Opponent', 'GF60/xGF60_even_Team',
                       'GF60/xGF60_pp_Opponent', 'GF60/xGF60_pp_Team',
                       'PEND60_Opponent', 'PEND60_Team',
                       'PENT60_Opponent', 'PENT60_Team',
                       'xGA60/FA60_even_Opponent', 'xGA60/FA60_even_Team',
                       'xGA60/FA60_pk_Opponent', 'xGA60/FA60_pk_Team',
                       'xGF60/FF60_even_Opponent', 'xGF60/FF60_even_Team',
                       'xGF60/FF60_pp_Opponent', 'xGF60/FF60_pp_Team',
                       'days_rest_home', 'days_rest_away',
                       'home_adj_fsv', 'away_adj_fsv']

    non_scaled = ['elo_prob']
    dummies = ['home_b2b', 'away_b2b']

    # Switch it over -> Don't want to overwrite anything
    df_scaled = df[continuous_vars + non_scaled + dummies]

    # Scale only continuous vars
    scaler = StandardScaler().fit(df_scaled[continuous_vars])
    df_scaled[continuous_vars] = scaler.transform(df_scaled[continuous_vars])

    # Save Scaler
    pickle.dump(scaler, open("team_scaler.pkl", 'wb'))

    return df_scaled[continuous_vars + non_scaled + dummies].values.tolist()

```

I then randomly split the data into a training and test set using our full dataset (which as
mentioned earlier is 2008-2009 to 2017-2018). I chose to use 75% of the data for training and 25% for testing.
After that we can then fit our model using the training data and test how it does on the training set.

I chose to fit both models using bagged logistic regressions. Well, I didn't really choose
that off the bat. I tried a bunch of different techniques for each model (xgBoost, Random
Forest...etc.) but at the end of the day a bagged logistic regression worked best for both.
I found that interesting. 

[Bagged](https://en.wikipedia.org/wiki/Bootstrap_aggregating) here means that it uses the 
concept of Bagging. Bagging is when random subsets of a dataset are used to create many different models. 
We then use the average prediction of all these models as our final answer. So here we 
made many different logistic regressions using random subsets of the training data. Then each 
of these logistic regressions combined (their outputs are averaged) to form one model. 

We can now fit and test each model using the code below:

```python
def train_test_clf(clf_type, features, labels):
    """
    Train the data on specific classifier. Then test how it does. 

    :param clf_type: Type of Classifier we are using
    :param features: Vars for model
    :param labels: What we are predicting

    :return: Trained Classifier
    """
    # 75/25 Split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.25, random_state=36)

    # Fix Data
    features_train, labels_train = np.array(features_train), np.array(labels_train).ravel()
    
    # Set up CV
    param_grid = {'n_estimators': range(2, 22, 2)}
    clf_type = BaggingClassifier(LogisticRegression(penalty='l1', solver='liblinear', random_state=36), random_state=36)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf_type, param_grid=param_grid, cv=5, verbose=2)
    
    # Actually fit model
    cv_clf = cv_clf_type.fit(features_train, labels_train)
    
    # Evaluate how the model performs on the etst set
    test_probs = clf.predict_proba(features_test)
    print("Log Loss: ", round(log_loss(labels_test, test_probs), 4))
    print("Accuracy:", round(accuracy_score(labels_test, [round(prob[1]) for prob in test_probs]), 4))

```

This function can fit both the player and team models but, as mentioned before, we want to combine
their outputs into one model. Each by themself is okay, but the best way to handle this 
is to leverage the strength of both models and combine them into one. 

Meta-Classifier
--------------- 

To fit the meta-classifier we first need to fit both models. Using this we then generate the
probabilities for each model for each game in our set. Then, using the same training/testing
sets as before, we fit a logistic regression with the features being the team and player model
outputs. A logistic regression will help us get the correct weight for each model relative
to each other.

So the features (fit using a logistic regression) for the meta-classifier are:

* Home Team Prob% - Team Model
* Home Team Prob% - Player Model


The evaluation metrics spit out for each model on the Test set are:

| Model         | Accuracy      | Log Loss  |
| ------------- |:-------------:|:---------:|
| Team          | 57.97%        | .6751     |
| Player        | 58.56%        | .677      |
| Meta          | 58.25%        | .6738     |


The main number here is log loss. Accuracy is important but log loss tells us how well
it does in terms of the probabilities generated and not just whether or not it was correct. 
And, it clearly is the best for the meta-classifier. Interestingly the team model performs
better then the player model in log loss (though the opposite is true for accuracy). Each
model also performs better than just assuming the home team wins (which occurs 54.8% of the
times). 


Conclusion
----------

In this post we created a model to predict the outcome of a NHL game. Two models were created:
A model with broader team statistics, and a model focusing on just the players for both teams.
We then created an ensemble model that fit a logistic regression over the output of both models.
This final model (and the individual sub-models) perform better than our baseline model (of
assuming the home team wins). 

There are couple of limitations to my model. The first is it is relatively simple (in my opinion).
There aren't really too many features used for both sub-models. One could create much larger feature
sets for both sub-models which may uncover more hidden patterns. This leads into the second
criticism, mainly being that the performance is nothing to write home about. To my knowledge 
(please tell me if I'm wrong), the only publicly available game prediction model with the
test evaluations recorded is Emmanuel Perry's model. While I don't think I could ever achieve
the success of his model (he is very smart), it does highlight some the weaknesses here. Mainly being
the smaller feature sets and simple model designs. For example, I could have used stacking
to increase the performance of each sub-model. I thought about it but I chose to keep it 
simple. It would be a different story if I planned to actually try to gamble using this 
model but I don't plan on it. 


**The code and data used to create the model (Only the final datasets are up there, the initial
data used to create everything is too much) for the model can be found [here](https://github.com/HarryShomer/NHL-Prediction-Model). 
  
<br><br> 
**Footnotes:** 
<br><br>
<a name="footnote1">1</a>. I chose to weight previous seasons using the weights 5-4-3-2 (with 5
being the nearest season). I also chose to regress using 2000 shots. 
<br><br>
<a name="footnote2">2</a>. For forwards the weights are 12-4-3 and for defensemen it's 12-5-3.
Forwards are regressed 540 minutes while for defensemen it's 660. 
<br><br>
<a name="footnote3">3</a>. For forwards the weights are 9-1-0 and for defensemen it's 17-2-0.
Forwards are regressed 370 minutes while for defensemen it's 710. 
<br><br>


