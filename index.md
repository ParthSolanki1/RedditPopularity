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
