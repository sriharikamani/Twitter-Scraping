##############################################################################################################################################
# Purpose       : Sentiment  Analysis  on  Boeing  737  Max  8  Crash
#               : Program to analyze tweets using the techniques WORDCLOUD, Frequent Tern, bi-gram,Dendogram analysis 
#                 and captured  the insights and sentiments of the global audience
##############################################################################################################################################
#
rm(list=ls(all=T)) 
cat("\014")
#
# Loading the required packages
library(data.table)
library(dplyr)
library(TTR)
library(SnowballC)
library(tm)
library(ggplot2)
library(RColorBrewer)
library(wordcloud)
library(wordcloud2)
library(topicmodels)
library(data.table)
library(stringi)
library(qdap)
library(dplyr)
library(rJava)
library(sentiment)
library(sentimentr)
library(dendextend)
library(RWeka)
library(syuzhet)
library(lubridate)
library(scales)
library(reshape2)
library(plotly)
library(dendroextras)
library(RWeka)
library(dendextend)
library(cluster)    # clustering 
library(clValid)

##############################################################################################################################################
#Setting the working directory
##############################################################################################################################################
setwd("C:/Users/SRIHARI/Documents/My Data/Twitter/Boeing 737 max 8")

# Load the dataset
data       <- read.csv("Boeing 737 Max 8.csv",header = FALSE)

####################################################################################################################
#  >>>>>  D A T A    C L E A N  U P   A C T I V I T I E S      >>>>>>>>>>>>>>>                                     #
####################################################################################################################

myStopWords       <- c((stopwords('english')),c("rt", "the","with","use", "used", "via", "number ","amp","two",
                                                "www","says","will","que","157","what","act","com","dot","need",
                                                "new","news","los","percent","pictwitter","know",""))

data$V2           <- genX(data$V2, " <", ">")   # Remove character string between < >  
removeURL         <- function(x) gsub("http[^[:space:]]*", "", x) 
removeSingle      <- function(x) gsub(" . ", " ", x)   
removefirstLetter <- function(x) gsub("^b", " ", x)  
removeReTweets    <- function(x) gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", x)
removeReference   <- function(x) gsub("@\\w+", "", x) 

####################################################################################################################
# Function to clean the corpus                                                                                     #
####################################################################################################################

clean_corpus <- function(clean_corpus){
  
  clean_corpus <- tm_map(clean_corpus, content_transformer(removefirstLetter)   )
  clean_corpus <- tm_map(clean_corpus, content_transformer(removeReTweets))
  clean_corpus <- tm_map(clean_corpus, content_transformer(removeReference))
  clean_corpus <- tm_map(clean_corpus, content_transformer(tolower))
  clean_corpus <- tm_map(clean_corpus, content_transformer(replace_abbreviation))
  #corpus <- tm_map(clean_corpus, content_transformer(replace_number))
  clean_corpus <- tm_map(clean_corpus, content_transformer(replace_contraction)	)
  clean_corpus <- tm_map(clean_corpus, content_transformer(replace_symbol)	)
  clean_corpus <- tm_map(clean_corpus, removePunctuation)
  clean_corpus <- tm_map(clean_corpus, removeWords, myStopWords)
  clean_corpus <- tm_map(clean_corpus, content_transformer(removeURL))
  clean_corpus <- tm_map(clean_corpus, content_transformer(removeSingle))
  clean_corpus <- tm_map(clean_corpus, stripWhitespace)
  #  clean_corpus <- tm_map(clean_corpus, stemDocument)
  return(clean_corpus)
}

####################################################################################################################
# Function to complete stemmed tweets                                                                              #
####################################################################################################################

stemCompletion2 <- function(x,dictionary) {
  x <- unlist(strsplit(as.character(x)," "))
  x <- x[x !=""]
  x <- stemCompletion(x, dictionary = dictionary)
  x <- paste(x, sep="", collapse=" ")
  PlainTextDocument(stripWhitespace(x))
}

####################################################################################################################
# Make tokenizer function                                                                                          #
####################################################################################################################

tokenizer <- function(x) 
  NGramTokenizer(x, Weka_control(min = 2, max = 2))

####################################################################################################################

tweets_corpus   <- VCorpus(VectorSource(data$V2)) # Convert tweets data into corpus
clean_corp      <- clean_corpus(tweets_corpus)
clean_corp_Copy <- clean_corp                     # Make a copy of cleaned corpus for future use 
clean_corp      <- tm_map(clean_corp, stemDocument)
clean_corp      <- lapply(clean_corp, stemCompletion2, dictionary=clean_corp_Copy)
clean_corp      <- VCorpus(VectorSource(clean_corp))

####################################################################################################################
# Create Document-term matix & Term-document matrix from the cleaned corpus                                        #
####################################################################################################################

boeing_737_Max_8_dtm <- DocumentTermMatrix(clean_corp)
boeing_737_Max_8_tdm <- TermDocumentMatrix(clean_corp)

####################################################################################################################
# Frequent term analysis   - Most frequent 50 terms used in the tweets                                             #
####################################################################################################################

freq.terms <- findFreqTerms(boeing_737_Max_8_tdm, lowfreq = 50)
freq.terms 
####################################################################################################################
# Term Frequency Chart                                                                                             #
####################################################################################################################

term.freq <- rowSums(as.matrix(boeing_737_Max_8_tdm))
term.freq <- subset(term.freq, term.freq > 75)
df        <- data.frame(term = names(term.freq), freq= term.freq)
ggplot(df, aes(reorder(term, freq),freq)) + theme_bw() + geom_bar(stat = "identity",fill = "red")  + coord_flip() +labs(list(title="Term Frequency Chart", x="Terms", y="Term Counts")) 

# Removing few words which are not required

# This action restores the corpus. Use if required

#a<- list()
#for (i in seq_along(clean_corp)) {
#  a[i] <- gettext(clean_corp[[i]][[1]]) #Do not use $content here!
#}
#a <- unlist(a) 
#clean_corp <- Corpus(VectorSource(a)) 

#clean_corp   <- tm_map(clean_corp,gsub,pattern = 'los',replacement = 'loss') 
#clean_corp   <- tm_map(clean_corp, stripWhitespace) 
####################################################################################################################
# Rebuild the term frequency chart                                                                                 #
####################################################################################################################

########
myStopWords <- c((stopwords('english')),c("what","twitter","tras",""))
clean_corp  <- tm_map(clean_corp,removeWords , myStopWords) 
clean_corp  <- tm_map(clean_corp, stripWhitespace)
#clean_corp  <- tm_map(clean_corp, content_transformer(gsub), pattern = "boein", replacement = "boeing")
#######

boeing_737_Max_8_tdm <- TermDocumentMatrix(clean_corp)
term.freq <- rowSums(as.matrix(boeing_737_Max_8_tdm))
term.freq <- subset(term.freq, term.freq > 60)
df        <- data.frame(term = names(term.freq), freq= term.freq)

freq.terms <- findFreqTerms(boeing_737_Max_8_tdm, lowfreq = 60)
freq.terms
ggplot(df, aes(reorder(term, freq),freq)) + theme_bw() + geom_bar(stat = "identity",fill = "red")  + coord_flip() +labs(list(title="Term Frequency Chart", x="Terms", y="Term Counts")) 

##### barplot

barplot(term.freq,,
        las=2,cex.names = 1.0, 
        col = rainbow(50),
        main = "Term Frequency Plot",xlab = "") #names(term.freq))

####################################################################################################################
# Inferences based on term-frequency chart                                                                         #
####################################################################################################################
#
# Plotted with most frequent terms whose frequency > 75 
# Dominated by terms Boeing 737 MAX followed by crash and airlines name, Ethiopian
# When it comes to Boeing 737 MAX most of the people interested talking planes are grounded. 
# Words like "China" ,"Indonesia" indicates, People talking 737 grounded in these countries
# People are discussing crash similarities with Lion airlines 737 and also the design of the aircraft
#
####################################################################################################################
# Word Cloud Analysis                                                                                              #
####################################################################################################################

word.freq <-sort(rowSums(as.matrix(boeing_737_Max_8_tdm)), decreasing= F)
pal<- brewer.pal(8, "Dark2")
wordcloud(words = names(word.freq), 
          freq = word.freq, min.freq = 50, 
          random.order = F, 
          colors = pal, 
          max.words = 1500,
          cale = c(5,0.3),
          rot.per = 0.3)

temp <- data.frame(names(word.freq),word.freq)

names(temp) = c('word','freq')
wordcloud2(temp,
           size = 1.0,
           shape = 'circle',
           minSize = 2)                    # Shapes are circle, star, triangle, pantagon

####################################################################################################################
# Inferences based on word cloud analysis                                                                         #
####################################################################################################################
# Most discussed words are about Boeing 737 MAX crash, Ethiopian, ground
# This indicates that when its comes to Boeing 737 MAX, "crash" and "air craft grounded" are the 
# hottest topics on the social media
# China and Indonesia ordered grounding of flights is another hot topic in the social media
# New word "suspend" is the hot topic in the social media
# People are discussing crash similarities with Lion airlines 737 and also the design of the aircraft
#
####################################################################################################################
# Text Mining. Clustering analysis                                                                                 #
####################################################################################################################
#####################################################################################################################
# Dendrogram -	Clustering by Term Similarity on TDM                                                               #
####################################################################################################################
# Before building Dendrogram from term document matrix, need to limit the number of words in 
# TDM to adjust the sparsity of the TDM, so that the generated Dendrogram is not cluttered and 
# is easily interpretable.
#############################
#
# Removing the sparse terms
boeing_737_Max_8_tdm_2 <- removeSparseTerms(boeing_737_Max_8_tdm, sparse=0.975)

# Create matrix
tdm_dendro <- as.matrix(boeing_737_Max_8_tdm_2)

# Create dataframe
tdm_dendro_df <- as.data.frame(tdm_dendro)

# Create distance matrix
tweets_dist <- dist(tdm_dendro_df, method = "euclidian")

# Create hc
hc <- color_clusters(hclust(tweets_dist),7,labels(hc$labels))

# Create hcd
hcd <- as.dendrogram(hc)

# Print the labels in hcd
labels(hcd)

# Change the branch color to red for "marvin" and "gaye"
hcd <- branches_attr_by_labels(hcd, c("marvin", "gaye"), "red")

# Plot hcd
hcd %>% set("labels_col", value = c("violetred4", "gold", "red","black","green","tomato3","cyan","violet","blue","coral1"),k=10) %>% 
  plot(main = "Dendrogram (Clustering by Term Similarity)",edgePar = list(col = 2:3, lwd = 2:1))
abline(h = 2, lty = 2)

# Add cluster rectangles 
rect.dendrogram(hcd, k = 7, border = "gold", xpd = FALSE, lower_rect = 0)

library("ggdendro")
ggdendrogram(hcd, theme_dendro = FALSE)

ggd1 <- as.ggdend(hcd)
ggplot(ggd1) 

rm(boeing_737_Max_8_tdm_2)
rm(tdm_dendro)
rm(tdm_dendro_df)
rm(tweets_dist)
rm(hc)
rm(hcd)

####################################################################################################################
# Dendrogram -	Clustering by Term Similarity on TDM                                                               #
####################################################################################################################
# The goal is to identify similar groups of documents and give top 15 words within each group                      #
####################################################################################################################
#
# Removing the sparse terms
#boeing_737_Max_8_dtm_2 <- removeSparseTerms(boeing_737_Max_8_dtm, sparse=0.975)

# Statistics

#dtms       = as.matrix(boeing_737_Max_8_dtm_2)
#dtms_freq  = as.matrix(rowSums(dtms))
#dtms_freq1 = dtms_freq[order(dtms_freq),]
#sd         = sd(dtms_freq)
#mean       = mean(dtms_freq)

#par(mfrow  = c(1,2))  

# Creating histogram and boxplot
#hist(dtms_freq,
#     main = "Histogram",
#     col = "green",
#     col.main = "dodgerblue")
#boxplot(dtms_freq, 
#        main = "Boxplot", 
#        col = "green",
#        col.main = "dodgerblue")

# Create matrix
#dtm_dendro <- as.matrix(boeing_737_Max_8_dtm_2)

#
#############
# Normalizing each document
#############
#
#for (i in 1:length(clean_corp)) {
#  DTM_m[i,] = as.matrix(DTM_m[i,])/norm(as.matrix(DTM_m[i,]), type ="F")
#}

# Create distance matrix
#tweets_dist <- dist(dtm_dendro, method = "euclidian")

# Clustering using "ward.D" method
#hc  = hclust(tweets_dist, method="ward.D")
#hcd <- as.dendrogram(hc)

# Find the value of K using Dunn Index
#k = 20
#mat = matrix(0, nrow = k, ncol = 2, byrow = TRUE)
#for (i in 1:k) {
#  members = cutree(hc, i)
#  dunn_index = dunn(clusters = members, Data = tweets_dist)
#  mat[i,1] = i
#  mat[i,2] = dunn_index
#}

# Plot number of clusters vs Dunn Index
#plot(mat, 
#     type = 'b',
#     xlab = "Number of Cluster", 
#     ylab = "Dunn Index",
#     pch = 16,
#     col = "red",
#     main = "Dunn's Index vs Number of clusters",
#     col.main = "dodgerblue")
#points(mat, col = "green")

# The selected value of K=4. Plot hcd
#K=4
#hcd %>% set("labels_col", value = c("violetred4", "black","green","tomato3"),k=K) %>% 
#  plot(main = "Dendrogram for DTM",leaflab = "none",edgePar = list(col = 2:3, lwd = 2:1))
#abline(h = 2, lty = 2)

# Add cluster rectangles 
#rect.dendrogram(hcd, k = K, border = "blue", xpd = FALSE, lower_rect = 0)

#clward1 = as.data.frame(cutree(hcd, K))

# Create list of clusters 
#cl = list()
#for (i in 1:K) {
#  cl[paste("cl_",i, sep = "")] = list(rownames(subset(clward1, clward1 == i)))
#}
#cl

# Creating corpuses for each cluster
#for (i in 1:K) {
#  name = paste("cl_corp_", i, sep = "")
#  assign(name, clean_corp[match(cl[[i]], names(clean_corp))])
#  print(name)
#} 

#Tdm = list()

# Creating a list of TDMs for each cluster
#for (i in 1:K) {
#  dtm_i = TermDocumentMatrix(get(paste("cl_corp_",i,sep="")))
#  tdm_i <- as.matrix(dtm_i)
#  Tdm[paste("cluster_",i,sep="")] = list(tdm_i)
#}

# Plot top 15 words in each cluster

#par(mfrow = c(2,2))
#for (i in 1:K) {
#  cl_m = as.matrix(Tdm[[i]])
#  barplot(sort(sort(rowSums(cl_m), decreasing = TRUE)[1:15], decreasing = FALSE),
#          las = 2,
#          horiz = TRUE,
#          decreasing = FALSE, 
#          main = paste("Top 15 words in cluster", i, sep = " "),
#          cex.main = 1.5,
#          cex.names = 1.0,
#          col = rainbow(10),
#          col.main = "tomato3")
#}

####################################################################################################################
# Sentiment Score based on NRC sentiment dictionary                                                                #
####################################################################################################################
#
myText <- data$V2
myText <- gsub("^b", " ", myText)                          # Remove first letter of all the tweets (as it is found that all tweets start with letter 'b'
myText <- gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", myText)  # Remove ReTweets 
myText <- gsub("@\\w+", "", myText)                        # Remove @ references from the tweets
myText <- stri_trans_tolower(myText)                       # Convert entire tweet text to lowercase
myText <- replace_abbreviation(myText)                     # Replace abbreviations with their full text equivalents
myText <- replace_contraction(myText)                      # Convert contractions back to their base words  
myText <- replace_symbol(myText)                           # Replace common symbols with their word equivalents
myText <- removePunctuation(myText)                        # Remove punctuations from the corpus
myText <- removeWords(words = myStopWords,myText)          # Remove Stopwords     
myText <- gsub("http[^[:space:]]*", "", myText)            # Remove the links (URLs)
myText <- gsub(" . ", " ", myText)                         # Remove Single letter words
myText <- stripWhitespace(myText)                          # Remove Extra Whitespace    
#myText <- stemDocument(myText)

# myText2V <- VCorpus(VectorSource(myText1))
pal<- brewer.pal(8, "Dark2")
wordcloud(words = myText, min.freq = 10, random.order = F, colors = pal, max.words = 200,
          scale = c(5,0.3),
          rot.per = 0.3)

###############
# Get the sentiments from cleaned up text and plot WordCloud
###############

mysentiment <- get_nrc_sentiment(myText)
SentiScore  <- data.frame(colSums(mysentiment[,]))
names(SentiScore) <- "Score"
SentiScore <- cbind("Sentiment" = rownames(SentiScore),SentiScore)
rownames(SentiScore) <- NULL

ggplot(data = SentiScore, aes(x = Sentiment, y = Score)) +
  geom_bar(aes(fill = Sentiment), stat = "identity") +
  theme(legend.position = "none") +
  xlab("Sentiment") + ylab("Score") + ggtitle("Sentiment Score for Boeing 737 max 8 Tweets on 11th March")

barplot(colSums(mysentiment),cex.names = 1.0,
        col = rainbow(10),
        main = "Sentiment Score for Boeing 737 max 8 Tweets on 12th March")
####################################################################################################################
# Inferences based Sentiment Score based                                                               #
####################################################################################################################
# Negative vibes dominating
#
####################################################################################################################
# Word Cloud Analysis of Positive and Negative Tweets                                                             
#####################################################################################################################

emo_737_tweets    <- sentiment(get_sentences(myText))
tweets            <- data.frame(myText)
tweets$sentiment  <- emo_737_tweets$sentiment   # attaching sentiment column to each tweet so that later can be useful tosegreate +ve and -ve
colnames(tweets)  <- c("Tweet","Sentiment")
positive_tweets   <- head(unique(tweets[order(emo_737_tweets$sentiment,decreasing = T),c(1,1)]),500)
negative_tweets   <- head(unique(tweets[order(emo_737_tweets$sentiment),c(1,1)]),500)            

rownames(positive_tweets) <- NULL
rownames(negative_tweets) <- NULL

write.table(positive_tweets$Tweet,file = "C:/Users/SRIHARI/Documents/My Data/Certification Courses/Rennes/Twitter/Boeing 737 max 8/Sentiments/positive.txt",sep="")
write.table(negative_tweets$Tweet,file = "C:/Users/SRIHARI/Documents/My Data/Certification Courses/Rennes/Twitter/Boeing 737 max 8/Sentiments/negative.txt",sep="")
tweets_corpus <- Corpus(DirSource(directory = "C:/Users/SRIHARI/Documents/My Data/Certification Courses/Rennes/Twitter/Boeing 737 max 8/Sentiments"))
Clean_corp      <- clean_corpus(tweets_corpus)
ctc_tdm         <- TermDocumentMatrix(Clean_corp)
ctc_matrix      <- as.matrix(ctc_tdm)
colnames(ctc_matrix) <- c("Negative Tweets","Positive Tweets")
comparison.cloud(ctc_matrix, max.words = 300,random_order = F)

#####################################################################################################################
# bigram analysis of tweets                                                                                          #
#####################################################################################################################

# So far, we have done our analysis on TDM built using single words (also called as unigrams). Now we will analyse 
# the tweets on TDM built on tokens containing two or more words. This can help us extracting useful phrases.

bigram_tdm <- TermDocumentMatrix(clean_corp, control = list(tokenize = tokenizer))
word.freq  <-sort(rowSums(as.matrix(bigram_tdm)), decreasing= F)
pal        <- brewer.pal(8, "Dark2")
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 74, random.order = F, colors = pal,
          max.words = 20)


####################################################################################################################
# Topic Modelling -  will now try to identify some latent/hidden topics in our tweets using LDA technique          #
####################################################################################################################
#
#################
# uni-gram tokens
#################

boeing_737_Max_8_dtm <- as.DocumentTermMatrix(boeing_737_Max_8_tdm)

rowTotals              <- apply(boeing_737_Max_8_dtm , 1, sum)
NullDocs               <- boeing_737_Max_8_dtm[rowTotals==0, ]
boeing_737_Max_8_dtm   <- boeing_737_Max_8_dtm[rowTotals> 0, ]

if (length(NullDocs$dimnames$Docs) > 0) {
  data <- data[-as.numeric(NullDocs$dimnames$Docs),]
}

lda   <- LDA(boeing_737_Max_8_dtm, k = 6) # find 5 topic
term  <- terms(lda, 6) 
term <- apply(term, MARGIN = 2, paste, collapse = ", ")

####################################################################################################################

#################
# bi-gram tokens
#################

boeing_737_Max_8_dtm <- as.DocumentTermMatrix(bigram_tdm)

rowTotals              <- apply(boeing_737_Max_8_dtm , 1, sum)
NullDocs               <- boeing_737_Max_8_dtm[rowTotals==0, ]
boeing_737_Max_8_dtm   <- boeing_737_Max_8_dtm[rowTotals> 0, ]

if (length(NullDocs$dimnames$Docs) > 0) {
  data <- data[-as.numeric(NullDocs$dimnames$Docs),]
}

lda   <- LDA(boeing_737_Max_8_dtm, k = 6) # find 5 topic
term  <- terms(lda, 6) 
term  <- apply(term, MARGIN = 2, paste, collapse = ", ")
term

####################################################################################################################
# Inferences: Main topics most discussed about Boeing737max8 crash
# Topic 1: Crash and planes grounded
# Topic 2: Model 737 suspended and compares with Lion air crash 
# Topic 3: Design change mandate for Boeing 737
####################################################################################################################

####################################################################################################################
#Sentiment Analysis of Tweets (classification by emotion)                                                          #
####################################################################################################################

#This function helps us to analyze some text and classify it in different types of #emotion: anger, disgust, #fear, joy, sadness, and surprise
class_emo = classify_emotion(myText, algorithm="bayes", prior=1.0)

# get emotion best fit
emotion = class_emo[,7]

# replace NA's by "unknown"
emotion[is.na(emotion)] = "unknown"

# The classify_polarity function allows us to classify some text as positive or negative
#class_pol = classify_polarity(data$V2, algorithm="bayes",pstrong=2.5,pweak=1.0,prior=1.0,verbose=FALSE)

class_pol = classify_polarity(myText, algorithm="voter")

# get polarity best fit
polarity = class_pol[,4]

# Create data frame with the results and obtain some general statistics
# data frame with results
sent_df = data.frame(text= myText, emotion=emotion, polarity=polarity, stringsAsFactors=FALSE)

# sort data frame
sent_df = within(sent_df, emotion <- factor(emotion, levels=names(sort(table(emotion), decreasing=TRUE))))
 
ggplot(sent_df, aes(x=emotion)) +
  geom_bar(aes(y=..count.., fill=emotion)) +
  scale_fill_brewer(palette="Dark2") +
  labs(x="emotion categories", y="number of tweets") +
  labs(title = "Sentiment Analysis of Tweets about Boeing 737_Max_8 on 12th March \n(classification by emotion)")

plot_ly(sent_df, x=~emotion,type="histogram",
        marker = list(color = c('grey', 'red','orange', 'navy','yellow'))) %>%
  layout(yaxis = list(title='Count'), title="Sentiment Analysis: Emotions")

#################
# classification by polarity
#################

ggplot(sent_df, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="Dark2") +
  labs(x="polarity categories", y="number of tweets") +
  labs(title = "Sentiment Analysis of Tweets about Boeing 737_Max_8\n(classification by polarity)")

plot_ly(sent_df, x=~polarity, type="histogram",
        marker = list(color = c('magenta', 'gold','lightblue'))) %>%
  layout(yaxis = list(title='Count'), title="Sentiment Analysis: Polarity")

#################
# Visualize the words by polarity
#################

sent_df <- sent_df %>%
  group_by(polarity) %>%
  summarise(pasted=paste(text, collapse=" "))

# remove stopwords
sent_df$pasted = removeWords(sent_df$pasted, stopwords('english'))

# create corpus
corpus_polarity        = Corpus(VectorSource(sent_df$pasted))
tdm_polarity           = TermDocumentMatrix(corpus_polarity)
tdm_polarity           = as.matrix(tdm_polarity)
colnames(tdm_polarity) = sent_df$polarity

###########
# comparison word cloud to plot neutral,positive and negative
###########

comparison.cloud(tdm_polarity, colors = brewer.pal(3, 'Dark2'),
                 scale = c(1.5,.5), random.order = FALSE, title.size = 1.5)

####################################################################################################################
# Inferences                                                                                                       #
####################################################################################################################
# People prefer to use more anger words about Boeing 737 max 8
# Lot of unknown emotions. Should be able to classify the polarity reasonably well
# Overall sentiment score is negative. Negative vibes in the social media
####################################################################################################################
