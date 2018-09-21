---
layout: page   # default
title: Projects
permalink: /projects/
---

[OffsideReview.com](http://offsidereview.com/)
=========================================

- OffsideReview is a website for modern National Hockey League statistics from the 2016-2017 season onwards for all Regular Season and Playoff games.
- Can query statistics for skaters, goalies, and teams with a variety of different filters (e.g. date, player, team).
- Each query makes an Ajax request to the server to retrieve the requested data.
- At 10 a.m. every morning the new games are scraped and then processed with the new data being added to the database. 


[hockey_scraper](https://github.com/HarryShomer/Hockey-Scraper)
==============

- Designed to allow one to scrape the Play-By-Play and Time-On-Ice tables off of the National Hockey League (NHL) API
and website.
- Can scrape the information for any preseason, regular season, and playoff game from the 2007-2008 season onwards. 
- Has functionality to scrape games by season, between a given date range, and by individual games.
- Available as a python package on pip as “hockey_scraper”.


[NHL Expected Goals Model](https://github.com/HarryShomer/xG-Model)
=========================

- Utilized machine learning techniques to create a model that predicts the probability of an unblocked shot of being a goal in the National Hockey League.
- Three different machine-learning classifiers were used to fit the same data and features, including: Logistic regression, Random Forest, and Gradient Boosting.
- All three models were evaluated using both log loss and by plotting it’s ROC curve and calculating its AUC score.
- The Gradient Boosting model performed best in both evaluation metrics followed by the Random Forest model and then the Logistic regression model.

