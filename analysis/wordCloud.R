# Add required libraries and packages
require(RSQLite)
require(dplyr)
library(wordcloud)
library(tm)
library(RColorBrewer)

# Create function
wordcloud <- function(subreddit) {
    sub <- subreddit
    max_words <- 75

    # Read the dataset using SQLite DB
    db <- src_sqlite('data.sqlite', create=F)
    dbsub <- db %>% 
             tbl('January2015') %>% 
             filter(subreddit==sub)      
    dbsub <- data.frame(dbsub)

    # Add all the required constraints for the corpus data
    corpus <- Corpus(VectorSource(dbsub$body))
    corpus <- tm_map(corpus, tolower)
    corpus <- tm_map(corpus, removePunctuation)
    corpus <- tm_map(corpus, removeWords, stopwords("english"))

    # Remove the following words and exclude them from the word cloud
    corpus <- tm_map(corpus, removeWords, c("like","cant", "dont", "im", "better", "think", "thats", "gt", "played", "go", "deleted", "time", "pretty", "got", "even", "will", "chance", "big", "love", "hes", "hope", "well", "last", "may", "3", "2", "oh", "youre", "look", "never", "just", "way", "see", "though", "thing", "still", "new", "best", "sure", "ever", "going", "make", "work", "really", "something", "things", "good", "get", "can", "maybe", "great", "different", "actually", "isnt", "doesnt", "use", "lot", "around", "take", "now", "fuck", "real", "two", "theyre", "mean", "someone", "years", "since", "fucking", "say", "made", "know", "find", "little", "point", "yeah", "said", "day", "getting", "looks", "many", "theres", "yes", "long", "old", "right", "shit", "used", "every", "bad", "first", "want", "can", "man", "probably", "everyone", "much", "buy", "hey", "thanks", "means", "open", "important", "top", "ive", "help", "less", "quite", "least", "also", "send", "tried", "bit", "usually", "havent", "decide", "soon", "youll", "usual"))
    dtm <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), stopwords = TRUE))
    
    # LSA - remove sparse items
    dtmsparse <- removeSparseTerms(dtm, 0.95)
    postwords <- as.data.frame(as.matrix(dtmsparse))
    totalwords <- data.frame(words = colnames(postwords), counts = colSums(postwords))  

    # Generate final word cloud              
    png(paste(subreddit,".png", sep=""))
    wordcloud(words = totalwords$words,freq=totalwords$counts,max.words = max_words,color = brewer.pal(4,"Dark"))   
    dev.off()
    print(subreddit)
}

# Call the function
wordcloud("datascience")