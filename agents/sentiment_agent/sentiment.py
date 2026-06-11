import os
import feedparser
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()


class SentimentAgent:

    def __init__(self):

        print("Loading Sentiment Agent...")

        self.classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )

        print("Sentiment Agent Ready")

    # --------------------------------
    # RSS NEWS FETCHER
    # --------------------------------
    def fetch_rss_news(self, ticker):

        feeds = [

            f"https://finance.yahoo.com/rss/headline?s={ticker}",

            "https://feeds.marketwatch.com/marketwatch/topstories/",

            "https://www.cnbc.com/id/100003114/device/rss/rss.html"

        ]

        headlines = []

        for url in feeds:

            feed = feedparser.parse(url)

            for entry in feed.entries[:5]:

                text = entry.title

                headlines.append(text)

        return headlines

    # --------------------------------
    # REDDIT SCRAPER
    # --------------------------------
    def fetch_reddit_posts(self, ticker):

        subreddits = [

            "stocks",
            "investing",
            "wallstreetbets",
            "StockMarket",
            "options"

        ]

        posts = []

        headers = {
            "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        ),
        "Accept": "application/json"
        }

        for subreddit in subreddits:

            url = (
                f"https://www.reddit.com/r/{subreddit}/search.json"
                f"?q={ticker}"
                f"&restrict_sr=1"
                f"&sort=relevance"
                f"&limit=5"
            )
            try:

                response = requests.get(
                url,
                headers=headers,
                timeout=10
            )
                print(f"\n{subreddit} STATUS:", response.status_code)
                if response.status_code != 200:
                   continue
                data=response.json()
                children=data["data"]["children"]
                for post in children[:5]:
                    title = post["data"]["title"]
                    print(title)
                    posts.append(title)

            except Exception as e:
                print(f"Reddit error: {e}")

        return posts

    # --------------------------------
    # SENTIMENT ANALYSIS
    # --------------------------------
    def analyze_sentiment(self, ticker):

        # Fetch RSS financial news
        articles = self.fetch_rss_news(ticker)

        # Fetch Reddit discussions
        reddit_posts = self.fetch_reddit_posts(ticker)

        # Combine all text
        all_text = articles + reddit_posts

        if not all_text:

            return {
                "ticker": ticker,
                "sentiment": "neutral",
                "score": 0.0
            }

        scores = []

        for text in all_text:

            result = self.classifier(text)[0]

            label = result['label']
            score = result['score']

            # FinBERT labels
            if label.lower() == "positive":

                scores.append(score)

            elif label.lower() == "negative":

                scores.append(-score)

            else:

                scores.append(0)

        average_score = sum(scores) / len(scores)

        # Final sentiment label
        if average_score > 0:

            sentiment = "positive"

        elif average_score < 0:

            sentiment = "negative"

        else:

            sentiment = "neutral"

        return {

            "ticker": ticker,

            "sentiment": sentiment,

            "score": round(float(average_score), 4),

            "news_articles": len(articles),

            "reddit_posts": len(reddit_posts),

            "total_items_analyzed": len(all_text)

        }