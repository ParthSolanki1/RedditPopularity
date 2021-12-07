# Predicting Popularity of Reddit Posts and Comments

In a society where the success of advertisements is determined by the amount of interaction that a post generates, it is very important to understand when and where to post or comment in order to garner the most interactions. Reddit is a social media site with a huge user base that is constantly posting and commenting. Within this social media site, users have the option of "upvoting" or "downvoting" posts and comments which, in turn, determines the score of that particular post or comment. In this research project, I hope to use data science and machine learning methods to try and understand what factors make a reddit post or comment popular or if it is even possible to do so. I hope to explore how straightforward factors like time and date of posting, but also more complicated factors like sentiment and content affect the popularity of a post. The implications of finding factors that affect reddit post popularity would include being able to optimize the posting of posts and comments to garner the most attention as well as getting a window into the human mind to see what may grab their attention.

Some Research questions I hope to answer are:

- How can one optimize their comment in order to receive a maximal score? (Main Question)
- When is the best time to post a comment or post?
- How does the tone of the comment (ie. whether it is positive or negative) affect comments?
- Is it possible to determine the popularity of a post by it's content?

## Methods

### Preprocessing Data

The data I used for both the posts and comments is the provided Text database. Due to limitations of my computer, which has very low ram as well as is very slow, I could not load the entirety of the files. As a result, I took chunks of 100 000 in which I had randomly sampled data until I got about 100 000 posts and 100 000 comments. This means the data retains the randomness of the original sample and is therefore a good representative set of the original dataset. Prior to starting the analysis, I had hypothesized that it is likely that the day of the week and time of the week you post play a huge role in the success of the post. As a result, I converted the created_utc to a day of the week and hour of day. To test my hypothesis, I plotted both those values for the top 10 comments and posts from each subreddit and noticed an interesting pattern: 

![image](https://user-images.githubusercontent.com/40770286/144957309-754d9024-a0ac-4934-84a3-349b101e3544.png)

![image](https://user-images.githubusercontent.com/40770286/144957372-e20d82b3-5aec-45d2-b685-6433bb5b782f.png)

![image](https://user-images.githubusercontent.com/40770286/144957397-c6278677-fa28-49ce-bbb7-e2fca023e382.png)

![image](https://user-images.githubusercontent.com/40770286/144957483-55c40a80-e1eb-41d0-be3d-42db37516010.png)

We can clearly see that the top posts and comments are not really affected by the day of week in which they were posted (as the counts are relatively similar), however, the time of day in which you post has a clearly defined pattern which indicates that top posts in subreddits are more likely to be posted during day time in the eastern timezone. This also indicates that a lot of reddit's users are in the western hemisphere. Since there is nearly an even split among the top posts and comments about the day of week in which it was posted, I had removed it as a feature. I had also removed all user information as that is unlikely to affect whether or not someone upvotes or downvotes. Finally, I had combined all the text and used CountVectorizer to find the counts of the top 2000 words as I will be using the bag of words technique for my models.

### Sentiment Analysis

Whenever there are viral moments on the internet it is very apparent that they are usually overwelhmingly positive or negative, as it is likely that polarizing topics may make each side defensive which leads to more interactions. This is an hypothesis I had made prior to this project. In order to test this, I had decided to use sentiment analysis techniques on the top posts from each subreddit. I used nltk library's SentimentIntensityAnalyzer with it's built in Vader lexicon on the top 10 posts and comments from each subreddit. I then took the mean of it's positive, negative, and neutral score and saw the following results:

mean of positive score = 0.11451042780748663

mean of negative score = 0.07247005347593583

mean of neutral score = 0.8114122994652406

This has dispoven my initial hypothesis, as clearly the top posts are overwhelming neutral. This initially took me by surprise. However, upon further examination, this makes a lot of sense since polarizing topics may recieve more downvotes which counts as a -1 on the score.

### Normalizing the Scores

Subreddits vary quite heavily in terms of the amount of active users it has. Some subreddits (ie. askReddit) are a lot more popular than others. As a result, simply posting on those popular subreddits may give more upvotes, but relatively speaking they may not be as popular as something else with less upvotes in a smaller subreddit. In order to account for this, some normalization needs to be done to the scores. Therefore, I replaced the score metric with what oercentile that score is in within its subreddit.

### Models

In this project I have used 3 classification models to try and classify whether or not a post or comment will be popular. In order to do this we much have some definition of "popular". For the purposes of this research project, I had defined being popular as being over the 50th percentile (ie. that post is greater than 50% of other posts in the same subreddit), and unpopular as being under the 50th percentile.

I also applied PCA to be able to visualize how the posts and comments look on the 2D level and whther or not there exists some visible pattern. These graphs were done using plotly and is therefore an interactive plot which is better to explore the data than a static graph. 

For posts:

<img width="994" alt="Screen Shot 2021-12-06 at 10 44 23 PM" src="https://user-images.githubusercontent.com/40770286/144962626-a4f36184-29d4-40e9-a518-af5973320426.png">

For comments:

<img width="956" alt="Screen Shot 2021-12-06 at 10 48 45 PM" src="https://user-images.githubusercontent.com/40770286/144962776-1c4e469d-98d7-4fba-b1a0-95e1f7354c5f.png">

The yellow points are of popular posts and comments while the blue are of the unpopular. Notice, the points are very close together and there is no observable pattern.

#### Logistic Regression

This is the first of the 3 models I had implemented. I had utilized all of the features gathered in the data preprocessing as well as the sentiment scores for around 64 000 posts and 64 000 comments. As seen below, the training error was imporoving as the number of features increased, however, the test dataset resulted in poor results for any number of features. Having a 56% error is really bad because there are 2 classifiers so even by just guessing there is a 50% chance you'd get the classificaation correctly. Thus, this model is not extendable to outside the training set. This is likely due to overfitting or poor feature selection, which are both addressed in the next 2 classifiers. However, the fact that logisic regression did not fit well means that, geometrically, there is no linear boundary between the classes.

Model error graph for posts:

![image](https://user-images.githubusercontent.com/40770286/144964075-c21ada0a-f53d-43e7-9330-0c93d42b2387.png)

Model error graph for comments:

![image](https://user-images.githubusercontent.com/40770286/144964108-39eee839-3c1a-4362-ad9d-15fd329f5c32.png)

#### Random Forest

This is the next model I had implemented. After having seen that logistic regression simply did not work, I had thought it may have been an issue with feature selection. Hence, I decided to use the random forest classifier which is quite resistant to bad feature choices since if a feature does not have enough variance, the tree would simply not use it as a condition to make a split. However, much to my suprise, while the training error was steadily decreasing the testing error remained the same, which is a good indication of overfitting.


Model error graph for posts:

![image](https://user-images.githubusercontent.com/40770286/144964686-dea5c1b3-b2e1-41c4-a776-df91ea03da48.png)

Model error graph for comments:

![image](https://user-images.githubusercontent.com/40770286/144964701-acfe478e-c87d-4166-8201-a1b9330e7b57.png)

#### Naive Bayes

The final classsification model I decided to use was Nai
